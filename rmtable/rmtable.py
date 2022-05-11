import numpy as np
import astropy.table as at
import astropy.coordinates as ac
import astropy.units as au
import astropy.io.votable as vot
import astropy.io.fits as pf
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

__version__ = "1.2"

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


class RMTable(at.Table):
    """A class for holding tables of RMs and associated columns.
    Will have associated methods for reading, writing, outputting to various
    types
    """

    def __init__(self, data=None, *args, **kwargs):
        version = __version__

        # standard_columns_file = pkg_resources.resource_filename(
        #     "standard_data", f"column_standard_v{version}.yaml"
        # )

        self.standard = __standard__
        self.standard_columns = list(self.standard.keys())
        self.standard_dtypes = [
            self.standard[col]["dtype"] for col in self.standard_columns
        ]
        self.standard_limits = [
            self.standard[col]["limits"] for col in self.standard_columns
        ]
        self.standard_blanks = [
            self.standard[col]["blank"] for col in self.standard_columns
        ]
        self.standard_units = [
            self.standard[col]["units"] for col in self.standard_columns
        ]
        self.standard_ucds = [
            self.standard[col]["ucd"] for col in self.standard_columns
        ]

        self.entries = __entries__
        self.standard_rm_method = self.entries["rm_method"]
        self.standard_pol_bias = self.entries["pol_bias"]
        self.standard_telescope = self.entries["telescope"]
        self.standard_classification = self.entries["classification"]
        self.standard_flux_type = self.entries["flux_type"]
        self.standard_complexity_test = self.entries["complexity_test"]

        if data is not None:
            super().__init__(data=data, *args, **kwargs)
            self.add_missing_columns()
            self.verify_limits()
            self.verify_standard_strings()
        else:
            super().__init__(
                names=self.standard_columns,
                dtype=self.standard_dtypes,
                units=self.standard_units,
            )
        # Add ucds to meta of each column
        for col in self.standard_columns:
            self[col].meta["ucd"] = self.standard[col]["ucd"]

        self.meta["VERSION"] = version

        # These are for when extra columns might be added.
        # They point into the table where the columns can be found.
        self.units = self.standard_units.copy()
        self.ucds = self.standard_ucds.copy()
        self.size = len(self)

    def read(*args, **kwargs):
        """Reads in a table from a file."""
        return RMTable(at.Table.read(*args, **kwargs))

    def verify_limits(self):
        """This function checks that all numerical columns conform to the 
        standard for limits on allowed numerical values. Mostly important for 
        angles, as the standard uses [0,180) and not (-90,90].
        Non-conforming entries should be checked and fixed before incorporation 
        into the master catalog.
        """
        good = True  # Remains true until a non-conforming entry is found.
        for i in range(len(self.standard_columns)):
            if self.standard_columns[i] not in self.columns:  # ignore missing columns.
                continue
            if self.standard_limits[i] == "":  # ignore string columns
                continue
            data = self[self.standard_columns[i]]
            overmax = (
                data > self.standard_limits[i][1]
            ).sum()  # count how many outside of acceptable range
            undermin = (data < self.standard_limits[i][0]).sum()
            if overmax + undermin > 0:
                print(
                    "Column '{}' has {} entries outside the range of allowed values!".format(
                        self.standard_columns[i], overmax + undermin
                    )
                )
                good = False
        if good == True:
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
        for entry in self["pol_bias"]:
            if (entry not in self.standard_pol_bias) and (entry not in invalid_polbias):
                invalid_polbias.append(entry)
        if len(invalid_polbias) > 0:
            print(
                "The following non-standard polarization bias correction method(s) were found (at least once each):"
            )
            print(*invalid_polbias, sep="\n")

        invalid_telescope = []
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
        for entry in self["flux_type"]:
            if (entry not in self.standard_flux_type) and (entry not in invalid_flux):
                invalid_flux.append(entry)
        if len(invalid_flux) > 0:
            print(
                "The following non-standard flux measurement type(s) were found (at least once each):"
            )
            print(*invalid_flux, sep="\n")

        invalid_complexity_test = []
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

        if (
            len(
                invalid_methods
                + invalid_polbias
                + invalid_telescope
                + invalid_type
                + invalid_flux
            )
            == 0
        ):
            print("No problems found with standardized string entries.")

    def add_missing_columns(self):
        """Adds in any missing default columns, with their default (blank) values.
        Can be used to make a table compliant with the standard in terms of having
        all the columns.
        """
        missing_columns = [
            column for column in self.standard_columns if column not in self.columns
        ]
        for column in missing_columns:
            i = self.standard_columns.index(column)
            if self.standard_blanks[i] == None:
                warnings.warn(f"Missing essential column: {column}")
            self.add_column(
                at.Column(
                    data=[self.standard[column]["blank"]] * len(self),
                    name=column,
                    dtype=self.standard[column]["dtype"],
                    unit=self.standard[column]["units"],
                    meta={"ucd":self.standard[column]["ucd"]},
                )
            )

    def to_table(self):
        """Returns the table object."""
        return at.Table(self)


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
        sc = ac.SkyCoord(long, lat, frame="icrs", unit=(au.deg, au.deg))
        new_long = sc.galactic.l.deg
        new_lat = sc.galactic.b.deg
    else:
        sc = ac.SkyCoord(long, lat, frame="galactic", unit=(au.deg, au.deg))
        new_long = sc.icrs.ra.deg
        new_lat = sc.icrs.dec.deg

    return new_long, new_lat


def convert_angles(angles):
    """Converts an array of angles to follow the [0,180) degree convention
    used in the RMTable standard.
    Inputs: angles (array-like): angles (in degrees)
    Returns: array of angles (in degrees) in the range [0,180)
    """
    # Multiple of 180° that should be added to force all values to be positive
    n = np.ceil(np.abs(np.min(angles)) / 180)

    return np.mod(angles + 180 * n, 180)
