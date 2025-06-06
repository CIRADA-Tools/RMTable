# RMTable
This repository contains Python 3 code for reading/writing tables of rotation measures following the proposed RMTable standard, 
as well as the consolidated RM catalog maintained in RMTable format.

### [Documentation](https://github.com/CIRADA-Tools/RMTable/wiki)
Detailed documentation of the code package, including installation instructions and example code, is available on the wiki linked above.  

The paper describing the RMTable standard and the consolidated catalog can be found [here](https://ui.adsabs.harvard.edu/abs/2023ApJS..267...28V/abstract).  


The main code is in rmtable.py, which contains the class definition for an 
RMTable, which contains all the columns in the RM Standard and automatically
fills in blank columns with default values.

Some code demonstrating the use of RMTables can be found in [example/example.py](https://github.com/CIRADA-Tools/RMTable/blob/master/example/example.py).

***

The most current version of the RMTable standard can be found in these documents:  
[RMTable standard columns](https://github.com/CIRADA-Tools/RMTable/raw/master/docs/Column_definitions.pdf)  
[Standard strings](https://github.com/CIRADA-Tools/RMTable/raw/master/docs/Standard_entries.pdf)  



***

### Catalog

The consolidated catalog, which currently contains 59 530 RMs from 46 catalogs, is available in the consolidated_catalog_* files above or these links: [FITS format](https://github.com/CIRADA-Tools/RMTable/raw/master/consolidated_catalog_ver1.2.0.fits.zip) [TSV format](https://github.com/CIRADA-Tools/RMTable/raw/master/consolidated_catalog_ver1.2.0.tsv.zip) and [VOTable format](https://github.com/CIRADA-Tools/RMTable/raw/master/consolidated_catalog_ver1.2.0.xml.zip)  
This catalog is provided on an as-is basis; there may be uncaught transcription errors in adapting the published catalogs into the RMTable catalog format. It's also known that some of the published values/sources are unphysical (negative Stokes I, fractional polarizations outside of \[0,1), unrealistic spectral indices, etc). Users should use their discretion when selecting sources in the catalog to use. Please see Section 3.2 of the [paper](https://ui.adsabs.harvard.edu/abs/2023ApJS..267...28V/abstract) for more suggestions on catalog usage.

The DOI for the current version of the catalog (ver1.2.0) is [10.5281/zenodo.10963566](https://zenodo.org/records/10963566). The DOI for all versions of the catalog is [10.5281/zenodo.6702842](https://zenodo.org/record/6702842).


The list of individual catalogs/papers that have been incorporated into the consolidated catalog, with some notes on how they were adapted, can be found [here](https://github.com/CIRADA-Tools/RMTable/raw/master/docs/Catalog_notes.pdf).

***

### Suggestions for catalog authors (or those converting older catalogs)
Given the large number of columns in the standard, it may seem to potential RM catalog authors that the process of generating a catalog in RMTable format could be more effort than is merited. The majority of the columns defined in the standard are optional and can be omitted or left blank without creating problems, although every column that is included increases the value of the catalog to future users. The key minimum elements that must be adhered to follow the RMTable standard are twofold: first, the standard columns that **are** included must use the naming convention and units of the standard (to avoid users being unable to combine catalogs, or combining catalogs with inconsistent units); second, any columns added that are outside the RMTable standard must not have a name conflict with any of the defined standard columns (e.g., a column labelled ``b'' would conflict with the Galactic Latitude column in RMTable). As long as those two conditions are satisfied, catalog authors have the freedom to choose how much effort they invest into including more of the standard columns. Using this package will automatically ensure both conditions are satisfied.


Conversions of new catalogs into the RMTable format for inclusion into the consolidated catalog are very welcome. If you are interested in contributing a catalog to the consolidated catalog, or find any errors with the catalogs already included, please contact me.

Cameron Van Eck (cameron.vaneck (at) anu.edu.au)

***

### Checklist for updating catalog (for maintainers):
 - Check new individual catalogs for suitability: load in Python, check catalog values (`table.verify_columns()`, `table.verify_limits()`, `table.verify_standard_strings()`, `table.enforce_column_formats()`, manual inspection of columns to check units). Save as .fits table into `individual_catalogs/` directory.
 - Update any existing catalogs as needed (e.g., updating `catalog_name` columns on papers that now have proper bibcodes)
 - Update `CATALOG_HISTORY` file to describe new catalogs, changes. Updates `docs/Catalogs_notes.tex` and recompile.
 - If standard strings need updating to include new values, modify `rmtable/entries_standard_v<X>.json`. Update `docs/Standard_entries.tex` and recompile.
 - Run Consolidate catalogs.ipynb notebook and save new consolidated catalog.
 - Git move old catalog .zip files to `old_catalogs/`
 - Rename new catalogs with correct version number, create .zip files.
 - Commit changes (changes to individual catalogs, new consolidated catalogs).
 - Submit new consolidated catalog to Zenodo.
 - udpate DOI and version number in README.md.
