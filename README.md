# RMTable
Code to read/write tables of rotation measures following the proposed standard, in Python 3.

Current working document for the RM Standard definition:

https://docs.google.com/document/d/1lo-W89G1X7xGoMOPHYS5japxJKPDamjEJ9uIGnRPnpo/edit

The main code is in rmtable.py, which contains the class definition for an 
RMTable, which contains all the columns in the RM Standard and automatically
fills in blank columns with default values.

Some code demonstrating the use of RMTables can be found in examples.py.

The prototype catalog is available in the consolidated_catalog_* files. This catalog is provided on an as-is basis; there may be uncaught transcription errors in adapting the published catalogs into the RMTable catalog format. It's also known that some of the published values/sources are unphysical (negative Stokes I, fractional polarizations outside of [0,1), unrealistic spectral indices, etc. Users should use their discretion when selecting sources in the catalog to use.

A list of papers included (and excluded) from the consolidated catalog can be found here:

https://docs.google.com/document/d/18jUUV0QmesTeZb0Ng_6MtdTouNxfWuaI59A5UpQY00s/edit?usp=sharing

