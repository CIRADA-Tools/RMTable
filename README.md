# RMTable
Code to read/write tables of rotation measures following the proposed standard, in Python.

Current working document for the RM Standard definition:
https://docs.google.com/document/d/1lo-W89G1X7xGoMOPHYS5japxJKPDamjEJ9uIGnRPnpo/edit

The main code is in rmtable.py, which contains the class definition for an 
RMTable, which contains all the columns in the RM Standard and automatically
fills in blank columns with default values.
Main methods (mostly self-explanatory):
write_FITS
read_FITS
write_tsv
read_tsv
to_numpy: converts data in RMTable object to numpy ndarray
input_numpy: takes numpy array, adds new rows to table, filling in blanks as needed
to_pandas: converts data to pandas data frame.
verify_limits: checks all numerical columns for unexpected values (i.e., angles outside of range)
verify_standard_strings: checks strings that have limited allowed values.
merge_tables

A prototype version of the master RMTable will be uploaded shortly. A list of papers included
(and excluded) from the master catalog can be found here:
https://docs.google.com/document/d/18jUUV0QmesTeZb0Ng_6MtdTouNxfWuaI59A5UpQY00s/edit?usp=sharing

