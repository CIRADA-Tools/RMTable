# RMTable
Code to read/write tables of rotation measures following the proposed standard, in Python 3.

### [Documentation](https://github.com/CIRADA-Tools/RMTable/wiki)
Detailed documentation, including installation instructions and example code, is available on the wiki linked above.  

***

The paper describing the RMTable standard and the consolidated catalog is currently being refereed. The current draft, with a description of the RMTable standard, can be found [here](https://www.dropbox.com/s/ebdnhad8vypx4cc/RMTable.pdf?dl=0).  
If that link dies, a slightly older working document for the RM Standard definition is [here](https://docs.google.com/document/d/1lo-W89G1X7xGoMOPHYS5japxJKPDamjEJ9uIGnRPnpo/edit).

The main code is in rmtable.py, which contains the class definition for an 
RMTable, which contains all the columns in the RM Standard and automatically
fills in blank columns with default values.

Some code demonstrating the use of RMTables can be found in examples.py.

***

The most current version of the RMTable standard can be found in these documents (for the moment, these are the same as the paper above):  
[RMTable standard columns](https://github.com/CIRADA-Tools/RMTable/raw/master/docs/Column_definitions.pdf)  
[Standard strings](https://github.com/CIRADA-Tools/RMTable/raw/master/docs/Standard_entries.pdf)  



***

### Catalog

The consolidated catalog, which currently contains 55 819 RMs from 42 catalogs, is available in the consolidated_catalog_* files above or these links: [FITS format](https://github.com/CIRADA-Tools/RMTable/raw/master/consolidated_catalog_ver1.0.0.fits.zip) [TSV format](https://github.com/CIRADA-Tools/RMTable/raw/master/consolidated_catalog_ver1.0.0.tsv.zip) and [VOTable format](https://github.com/CIRADA-Tools/RMTable/raw/master/consolidated_catalog_ver1.0.0.xml.zip)  
This catalog is provided on an as-is basis; there may be uncaught transcription errors in adapting the published catalogs into the RMTable catalog format. It's also known that some of the published values/sources are unphysical (negative Stokes I, fractional polarizations outside of \[0,1), unrealistic spectral indices, etc). Users should use their discretion when selecting sources in the catalog to use. Please see Section 3.2 of the [paper](https://www.dropbox.com/s/ebdnhad8vypx4cc/RMTable.pdf?dl=0) for more suggestions on catalog usage.

The list of individual catalogs/papers that have been incorporated into the consolidated catalog, with some notes on how they were adapted, can be found [here](https://github.com/CIRADA-Tools/RMTable/raw/master/docs/Catalog_notes.pdf).

***

Conversions of new catalogs into the RMTable format for inclusion into the consolidated catalog are very welcome. If you are interested in contributing a catalog to the consolidated catalog, or find any errors with the catalogs already included, please contact me.

Cameron Van Eck (cameron.van.eck (at) utoronto.ca)

