import numpy as np
from astropy.table import Table, Column, MaskedColumn
from astropy.io import votable as vot
from astropy.coordinates import SkyCoord
import astropy.units as au
import json
import warnings

try:
    import importlib.resources as importlib_resources
except ImportError:
    # In PY<3.7 fall-back to backported `importlib_resources`.
    import importlib_resources


# TODO chose format for standard def

"""
This module defines the RMTable class and all the methods necessary to read and write
RMTables into ASCII (tsv) or FITStable formats as well as convert new catalogs in the 
form of a numpy ndarray into new RMTables with the correct columns. Methods to convert
RMTables into numpy arrays or pandas dataframes are also included.
A full description of the RMTable standard is currently located at 
https://docs.google.com/document/d/1lo-W89G1X7xGoMOPHYS5japxJKPDamjEJ9uIGnRPnpo/edit
"""

__version__ = "1.3"

# Reading in standard here so they are read on import
# rather than on first use.
with importlib_resources.open_text(
    "rmtable", f"column_standard_v{__version__}.json"
) as f:
    __standard__ = json.load(f)
assert __standard__.pop("version") == __version__, "Version number mismatch"
# Define standard entries for strings:
with importlib_resources.open_text(
    "rmtable", f"entries_standard_v{__version__}.json"
) as f:
    __entries__ = json.load(f)
assert __entries__.pop("version") == __version__, "Version number mismatch"


class RMTable(Table):
    """A class for holding tables of RMs and associated columns.
    Will have associated methods for reading, writing, outputting to various
    types.
    This class inherits from astropy.table.Table. The convenience functions
    that work with Tables (e.g. astropy.table.vstack) can also be used on
    RMTables.
    """

    def __init__(self, data=None, *args, **kwargs):
        self.version = __version__

        self.standard = __standard__
        self.standard_columns = list(self.standard.keys())
        self.standard_dtypes = {
            col: self.standard[col]["dtype"] for col in self.standard_columns
        }
        self.standard_limits = {
            col: self.standard[col]["limits"] for col in self.standard_columns
        }
        self.standard_blanks = {
            col: self.standard[col]["blank"] for col in self.standard_columns
        }
        self.standard_units = {
            col: self.standard[col]["units"] for col in self.standard_columns
        }
        self.standard_ucds = {
            col: self.standard[col]["ucd"] for col in self.standard_columns
        }
        self.standard_descriptions = {
            col: self.standard[col]["description"] for col in self.standard_columns
        }

        self.entries = __entries__
        self.standard_rm_method = self.entries["rm_method"]
        self.standard_pol_bias = self.entries["pol_bias"]
        self.standard_telescope = self.entries["telescope"]
        self.standard_classification = self.entries["classification"]
        self.standard_flux_type = self.entries["flux_type"]
        self.standard_complexity_test = self.entries["complexity_test"]
        self.standard_ionosphere = self.entries["ionosphere"]

        super().__init__(data=data, *args, **kwargs)
        # Add ucds to meta of each column
        self._add_ucds()
        # Add descriptions to each column
        self._add_descriptions(init=True)
        # These are for when extra columns might be added.
        # They point into the table where the columns can be found.
        self._set_rmtab_attrs()
        # Replace any masked columns with fill values
        self._unmask()

    def _set_rmtab_attrs(self):
        self.meta["VERSION"] = self.version
        self.units = {col: self[col].unit for col in self.columns}
        self.ucds = {col: self[col].meta["ucd"] for col in self.columns}
        self.size = len(self)

    def _new_from_slice(self, slice_):
        # For some dumb reason, the OG _new_from_slice method
        # clears the meta attributes.
        ret = super()._new_from_slice(slice_)
        ret._add_ucds()
        ret._add_descriptions(init=True)
        ret._set_rmtab_attrs()
        return ret

    def _add_ucds(self):
        """Adds ucds to the meta of each column."""
        for col in self.columns:
            # Check if ucd has already been set
            if not "ucd" in self[col].meta or self[col].meta["ucd"] is None:
                # If not, set it from the standard
                if col in self.standard_columns:
                    if "ucd" in self[col].meta and self[col].meta["ucd"] is None:
                        # Issue a warning here if the ucd set to None
                        warnings.warn(
                            f"Empty ucd for column '{col}', replacing with standard '{self.standard[col]['ucd']}'"
                        )
                    self[col].meta["ucd"] = self.standard[col]["ucd"]
                else:
                    self[col].meta["ucd"] = None

    def _add_descriptions(self,init=False):
        """Adds descriptions to each column.
        init (bool): if true, disables warnings."""
        for col in self.columns:
            # Check if description has already been set
            if col in self.standard_columns and self[col].description is None:
                if not init:
                    warnings.warn(
                        f"Empty description for column '{col}', replacing with standard '{self.standard[col]['description']}'"
                    )
                self[col].description = self.standard_descriptions[col]

    def read(*args, **kwargs):
        """Reads in a table from a file."""
        table = RMTable(Table.read(*args, **kwargs))
        table.update_version()
        return table

    def _unmask(self):
        for col in self.columns:
            if type(self[col]) is MaskedColumn:
                if self[col].dtype.kind == "f":
                    fill_value = np.nan
                elif self[col].dtype.kind == "i":
                    fill_value = -2147483648
                elif self[col].dtype.kind == "S" or self[col].dtype.kind == "U":
                    fill_value = ""
                elif col in self.standard_blanks:
                    fill_value = self.standard_blanks[col]
                else:
                    warnings.warn(f"Could not find a fill value for {col} - using None")
                    fill_value = None
                try:
                    new_col = self[col].filled(fill_value=fill_value)
                except TypeError:
                    warnings.warn(
                        f"Could not fill '{col}' (dtype {self[col].dtype}) with '{fill_value}' - using None"
                    )
                    new_col = self[col].filled(fill_value=None)
                self[col] = new_col

    def write_votable(self, filename, *args, **kwargs):
        """Writes the table to a VOTable file."""
        super().write(filename, *args, **kwargs, format="votable")

    def write_tsv(self, filename, *args, **kwargs):
        """Writes the table to a tsv file."""
        super().write(filename, *args, **kwargs, format="ascii.tab")

    def read_tsv(filename, *args, **kwargs):
        """Reads a table from a tsv file."""
        table=RMTable.read(filename, *args, **kwargs, format="ascii.tab")
        table.update_version()
        return table


    @classmethod
    def input_numpy(
        cls, array, verbose=False, verify=True, keep_cols=[], coordinate_system="icrs"
    ):
        """Converts an input numpy array into an RM table object.
        Requires that array has named columns matching standard column names.
        Will automatically fill in missing columns with default values.
        Non-standard columns listed in keep_cols will be kept, otherwise they will be discarded.
        Input parameters: array (numpy ndarray): array to transform.
                          verbose (Boolean): report missing columns
                          verify (Boolean): check if values conform to standard.
                          keep_cols (list): List of extra columns to keep.
                          coordinate_system (str): name of coordinate frame for
                                                   ra and dec columns
                                                   (typically 'fk5' or 'icrs')
        """

        Nrows = array.size
        self = cls()

        # This function needs to:
        # Check that columns in ndarray have correct names, and no extra columns are present. (case senstitive?)
        # Check for missing essential columns?
        # Build new table, mapping correct input columns and adding appropriate blanks.
        # Generate second coordinate system if missing.

        # Check for missing keep_col entries, and stop if they're not present in the array.
        missing_keep_cols = [
            column for column in keep_cols if column not in array.dtype.names
        ]
        if len(missing_keep_cols) > 0:
            print("The following columns cannot be kept. Check spelling!")
            print(*missing_keep_cols, sep=",")
            raise RuntimeError()

        newtable = [None] * len(self.standard_columns)
        array_columns = array.dtype.names
        matching_columns = []  # Track 'good' columns in input array
        additional_columns = []  # Track 'useless' columns in input array
        missing_columns = list(
            self.standard_columns
        ).copy()  # Track columns not present in table
        for col in array_columns:
            if (
                col in self.standard_columns
            ):  # Put the matching columns in the correct places
                matching_columns.append(col)
                missing_columns.remove(col)
                newtable[self.standard_columns.index(col)] = array[col]
            elif col in keep_cols:
                pass
            else:
                additional_columns.append(col)

        # Report extraneous columns:
        if verbose:
            print("Incorporated columns:")
            print(*matching_columns, sep="\n")
            print()

            print("Unused columns (check for spelling/capitalization errors!):")
            print(*additional_columns, sep="\n")
            print()
        # Check frame and convert to ICRS if needed, overwriting the ra and dec columns:
        if ("ra" in matching_columns) and (coordinate_system != "icrs"):
            print(f"Converting coordinates from {coordinate_system}.\n")
            coords = SkyCoord(
                array["ra"], array["dec"], frame=coordinate_system, unit="deg"
            )
            array["ra"] = coords.icrs.ra.deg
            array["dec"] = coords.icrs.dec.deg

        # Check position columns, generate missing ones
        if ("ra" in matching_columns) and ("l" in matching_columns):
            pass  # Both present, all is well.
        elif ("ra" in matching_columns) and ("l" not in matching_columns):
            # calculate Galactic:
            long, lat = cls.calculate_missing_coordinates_column(
                array["ra"], array["dec"], True
            )
            newtable[self.standard_columns.index("l")] = long
            newtable[self.standard_columns.index("b")] = lat
            matching_columns.append("l")
            missing_columns.remove("l")
            matching_columns.append("b")
            missing_columns.remove("b")
        elif ("ra" not in matching_columns) and ("l" in matching_columns):
            # calculate Equatorial
            long, lat = cls.calculate_missing_coordinates_column(
                array["l"], array["b"], False
            )
            newtable[self.standard_columns.index("ra")] = long
            newtable[self.standard_columns.index("dec")] = lat
            matching_columns.append("ra")
            missing_columns.remove("ra")
            matching_columns.append("dec")
            missing_columns.remove("dec")
        else:
            print(
                "No position columns found in input table! Make sure RA/Dec or l/b columns haven't been lost!"
            )
            raise RuntimeError()

        # Create missing columns with blank values:
        print("Missing columns (filling with blanks):")
        for col in missing_columns:
            newtable[self.standard_columns.index(col)] = np.repeat(
                self.standard_blanks[col], Nrows
            )
            print(col)
        print()

        newrows = cls(
            data=np.array(newtable).transpose(),
            names=self.standard_columns,
            dtype=self.standard_dtypes.values(),
        )

        # Add kept columns:
        for column in keep_cols:
            newrows.add_column(array[column], column)

        if verify == True:
            newrows.verify_limits()
            newrows.verify_standard_strings()

        return newrows

    def add_column(
        self,
        col,
        index=None,
        name=None,
        rename_duplicate=False,
        copy=True,
        default_name=None,
        ucd=None,
        unit=None,
    ):
        """Adds a column to the table."""
        # Run the astropy add_column method
        ret = super().add_column(col, index, name, rename_duplicate, copy, default_name)

        # Add ucds/units if they are called in this function
        if hasattr(col, "name"):
            colname = col.name
        else:
            colname = name
        self[colname].unit = unit
        self.units[colname] = unit
        self[colname].meta["ucd"] = ucd
        self.ucds[colname] = ucd

        # Add ucds/units if they are in the column data
        if hasattr(col, "unit"):
            assert unit is None, "Cannot specify both unit and unit in column"
            self.units[colname] = col.unit
            self[colname].unit = col.unit
        if hasattr(col, "meta"):
            if "ucd" in col.meta:
                assert ucd is None, "Cannot specify both ucd and ucd in column"
                self.ucds[colname] = col.meta["ucd"]
                self[colname].meta["ucd"] = col.meta["ucd"]
        self._add_ucds()
        self._add_descriptions()
        return ret

    def add_columns(
        self, cols, indexes=None, names=None, copy=True, rename_duplicate=False
    ):
        """Adds multiple columns to the table."""
        ret = super().add_columns(cols, indexes, names, copy, rename_duplicate)
        for col in cols:
            if hasattr(col, "unit"):
                self.units[col.name] = col.unit
            if hasattr(col, "meta"):
                if "ucd" in col.meta:
                    self.ucds[col.name] = col.meta["ucd"]

        self._add_ucds()
        self._add_descriptions()
        return ret

    def remove_column(self, name):
        """Removes a column from the table."""
        return super().remove_column(name)

    def remove_columns(self, names):
        """Removes multiple columns from the table."""
        for name in names:
            del self.units[name]
            del self.ucds[name]
        return super().remove_columns(names)

    def verify_columns(self):
        """Checks columns against standard, reporting missing or additional columns."""
        if self.standard_columns != self.colnames:
            print(
                "Columns inconsistent with standard. Check if deliberate (i.e., verify these aren't misspellings)."
            )
            extra_columns = [
                column for column in self.columns if column not in self.standard_columns
            ]
            missing_columns = [
                column for column in self.standard_columns if column not in self.columns
            ]
            print("Extra columns: ", end="")
            print(extra_columns)
            print("Missing columns: ", end="")
            print(missing_columns)
            # print(set(self.table.colnames).symmetric_difference(set(self.standard_columns)))

    def verify_limits(self):
        """This function checks that all numerical columns conform to the
        standard for limits on allowed numerical values. Mostly important for
        angles, as the standard uses [0,180) and not (-90,90].
        Non-conforming entries should be checked and fixed before incorporation
        into the master catalog.
        """
        good = True  # Remains true until a non-conforming entry is found.
        for col in self.standard_columns:
            if col not in self.columns:  # ignore missing columns.
                continue
            if self.standard_limits[col] == "":  # ignore string columns
                continue
            data = self[col]
            overmax = (
                data > self.standard_limits[col][1]
            ).sum()  # count how many outside of acceptable range
            undermin = (data < self.standard_limits[col][0]).sum()
            if overmax + undermin > 0:
                print(
                    f"Column '{col}' has {overmax + undermin} entries outside the range of allowed values!"
                )
                good = False
        if good:
            print("All columns conform with standard.")

    def verify_standard_strings(self):
        """This function checks the standardized string columns that they
        conform to the currently defined standard options. This is not strictly
        enforced, as the standard options are certainly incomplete.
        If assembling a catalog, please check that non-conforming values are
        not the result of typos, and contact the standard curator to have new
        options added to the standard.
        """
        invalid_methods = []
        if "rm_method" in self.colnames:
            for entry in self["rm_method"]:
                if (entry not in self.standard_rm_method) and (
                    entry not in invalid_methods
                ):
                    invalid_methods.append(entry)
            if len(invalid_methods) > 0:
                print(
                    "The following non-standard RM method(s) were found (at least once each):"
                )
                print(*invalid_methods, sep="\n")

        invalid_polbias = []
        if "pol_bias" in self.colnames:
            for entry in self["pol_bias"]:
                if (entry not in self.standard_pol_bias) and (entry not in invalid_polbias):
                    invalid_polbias.append(entry)
            if len(invalid_polbias) > 0:
                print(
                    "The following non-standard polarization bias correction method(s) were found (at least once each):"
                )
                print(*invalid_polbias, sep="\n")

        invalid_telescope = []
        if "telescope" in self.colnames:
            for entry in self["telescope"]:
                for scope in entry.split(","):
                    if (scope not in self.standard_telescope) and (
                        scope not in invalid_telescope
                    ):
                        invalid_telescope.append(scope)
            if len(invalid_telescope) > 0:
                print(
                    "The following non-standard telescope(s) were found (at least once each):"
                )
                print(*invalid_telescope, sep="\n")


        invalid_type = []
        if "type" in self.colnames:
            for entry in self["type"]:
                if (entry not in self.standard_classification) and (
                    entry not in invalid_type
                ):
                    invalid_type.append(entry)
            if len(invalid_type) > 0:
                print(
                    "The following non-standard source classification(s) were found (at least once each):"
                )
                print(*invalid_type, sep="\n")


        invalid_flux = []
        if "flux_type" in self.colnames:
            for entry in self["flux_type"]:
                if (entry not in self.standard_flux_type) and (entry not in invalid_flux):
                    invalid_flux.append(entry)
            if len(invalid_flux) > 0:
                print(
                    "The following non-standard flux measurement type(s) were found (at least once each):"
                )
                print(*invalid_flux, sep="\n")

        invalid_complexity_test = []
        if "complex_test" in self.colnames:
            for entry in self["complex_test"]:
                if (entry not in self.standard_complexity_test) and (
                    entry not in invalid_complexity_test
                ):
                    invalid_complexity_test.append(entry)
            if len(invalid_complexity_test) > 0:
                print(
                    "The following non-standard complexity test type(s) were found (at least once each):"
                )
                print(*invalid_complexity_test, sep="\n")


        invalid_ionosphere = []
        if "ionosphere" in self.colnames:
            for entry in self["ionosphere"]:
                if (entry not in self.standard_ionosphere) and (
                    entry not in invalid_ionosphere
                ):
                    invalid_ionosphere.append(entry)
            if len(invalid_ionosphere) > 0:
                print(
                    "The following non-standard ionosphere correction type(s) were found (at least once each):"
                )
                print(*invalid_ionosphere, sep="\n")
        if (
            len(
                invalid_methods
                + invalid_polbias
                + invalid_telescope
                + invalid_type
                + invalid_flux
                + invalid_ionosphere
            )
            == 0
        ):
            print("No problems found with standardized string entries.")


    def verify_ucds(self):
        """Confirm that all UCDs follow the expected conventions.
        """
        for colname, ucd in self.ucds.items():
            check = vot.ucd.check_ucd(ucd)
            if not check:
                warnings.warning(f"{colname} has invalid ucd '{ucd}'")




    def add_missing_columns(self):
        """Adds in any missing default columns, with their default (blank) values.
        Can be used to make a table compliant with the standard in terms of having
        all the columns.
        """
        missing_columns = [
            column for column in self.standard_columns if column not in self.columns
        ]
        for col in missing_columns:
            if self.standard_blanks[col] == None:
                warnings.warn(f"Missing essential column: {col}")
            self.add_column(
                Column(
                    data=[self.standard_blanks[col]] * len(self),
                    name=col,
                    dtype=self.standard_dtypes[col],
                    unit=self.standard_units[col],
                    meta={"ucd": self.standard_ucds[col]},
                )
            )

    def to_table(self):
        """Returns the table object."""
        return Table(self)



    def update_version(self):
        """Updates read-in tables with older versions to the most recent version.
        This can't be done using the version number, since the version number
        doesn't survive the read-write process. Must look for column names.
        
        Changes to be made:
            v1.2: change interval column name to obs_interval
                  change catalog column name to catalog_name
            v1.1: Remove id, flagA/B/C_value, flagA/B/C_name columns
            v1.0: Add rmsf_fwhm column"""
            
        #v1.0 updates:
        if 'rmsf_fwhm' not in self.colnames:
                self.add_column(
                    Column(
                        data=[self.standard_blanks['rmsf_fwhm']] * len(self),
                        name='rmsf_fwhm',
                        dtype=self.standard_dtypes['rmsf_fwhm'],
                        unit=self.standard_units['rmsf_fwhm'],
                        meta={"ucd": self.standard_ucds['rmsf_fwhm']},
                    )
                )

        #v1.1 updates:
        if 'flagA_value' in self.colnames:
            self.remove_columns(['id','flagA_value','flagA_name',
                                           'flagB_value','flagB_name',
                                           'flagC_value','flagC_name'])
            
        #v1.2 updates:
        if 'catalog' in self.colnames:
            self.rename_column('catalog','catalog_name')
            self.rename_column('interval','obs_interval')

            
        

    @staticmethod
    def calculate_missing_coordinates_column(long, lat, to_galactic):
        """Calculate a new pair of coordinate columns (equatorial/galactic) given
        the other pair and specified direction. Assumes input columns are already
        in degrees.
        Uses astropy coordinates for the transform.
        Input parameters:
            long: longitude column (ra/l)
            lat: latitude column (dec,b)
            to_galactic  (Boolean): direction of calculation: True = Equatorial -> Galactic, False = Galactic -> Equatorial
        Outputs: two arrays, new_long and new_lat
        """
        if to_galactic:
            sc = SkyCoord(long, lat, frame="icrs", unit=(au.deg, au.deg))
            new_long = sc.galactic.l.deg
            new_lat = sc.galactic.b.deg
        else:
            sc = SkyCoord(long, lat, frame="galactic", unit=(au.deg, au.deg))
            new_long = sc.icrs.ra.deg
            new_lat = sc.icrs.dec.deg

        return new_long, new_lat

    @staticmethod
    def convert_angles(angles):
        """Converts an array of angles to follow the [0,180) degree convention
        used in the RMTable standard.
        Inputs: angles (array-like): angles (in degrees)
        Returns: array of angles (in degrees) in the range [0,180)
        """
        # Multiple of 180° that should be added to force all values to be positive
        n = np.ceil(np.abs(np.min(angles)) / 180)

        return np.mod(angles + 180 * n, 180)
