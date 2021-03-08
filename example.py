#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Example code demonstrating the use of the RMTable class.

Created on Fri Apr 26 09:42:09 2019
@author: cvaneck
"""

import numpy as np
import rmtable as RMT


#Reading in an RMTable from FITS:
catalog=RMT.read_FITS('VanEck2011_table.fits')
print(catalog)

#Get the list of columns present in table:
print(catalog.columns)

#Get number of RMs (2 methods):
print(catalog.size)
print(len(catalog))

#Access column(s):
print(catalog['rm'])
print(catalog['l','b','rm','rm_err'])

#Access row(s):
print(catalog[0:10])

#Extracting a subset of the catalog:
selection=np.logical_and(catalog['l'] > 90,catalog['l'] < 270)
#This is an array of booleans, that can be used to extract a portion of the catalog.
print(catalog[selection])
#Multiple selections are combined using numpy's logical_and function.


#If you prefer numpy arrays or pandas dataframes, convert to those:
print(catalog.to_numpy())
print(catalog.to_pandas())


#Saving a sub-catalog:
subcatalog=catalog[selection]
subcatalog.write_FITS('subcatalog.fits',overwrite=True)






#How to convert other tables containing RMs into RMTables.
#Step 1: Read the other table into a numpy array with named columns matching
#   the RMTable column names:
cat=np.genfromtxt('./VanEck2011.dat',encoding=None,dtype=None,
              delimiter=[6,6,3,3,5,4,3,3,5,4,5,3,5,3,7,5],
            names=['l','b','rah','ram','ra','decd','decm','dec','stokesI',
                   'polint','rm','rm_err','RMSynth','dRMsynth','NVSSRM','dNVSSRM'])
#Setting the column names correctly is important to get the data into the RMTable.
# Columns with incorrect names are ignored in the conversion process.
#Note that I have called the RA seconds column simply 'ra', and Dec arcseconds 
#   column simply 'dec'. This is because it's a pain in the ass to add columns
#   to numpy arrays, so it's easier to modify existing columns (see step 2) than
#   add new columns after.

#Step 2: do necessary unit conversions to match RMTable convention.
#In this example, converting fluxes from mJy in the input table to Jy in the RMTable.
cat['polint']=cat['polint']/1e3
cat['stokesI']=cat['stokesI']/1e3
#And converting sexigessimal coordinates to decimal degrees, and storing the the
#columns that already have the right names:
cat['ra']=15*(cat['rah']+cat['ram']/60.+cat['ra']/3600.)
dec_sign=[ 1 if x==' +' else -1 for x in cat['dec_sign'] ]
table.table['dec']=(cat['decd']+cat['decm']/60.+cat['dec']/3600.)*dec_sign
#Note that these steps overwrite the data in the array, so DO NOT RUN TWICE!

#Step 3: convert to RMTable. It will automatically identify which columns are
#   part of the standard and which are not, based on the column names.
table=RMT.input_numpy(cat,verbose=True,verify=True)
#If verbose=True, it will report which columns were used or ignored, and which
#   are missing and filled with blanks.
#If verify=True, it will check that the numerical values are as expected.
#This typically means things like angle conventions (i.e. polarization angles 
#   from [0,180)).

#Step 4: add any information that wasn't in the input table. Most important is
#   the source (catalog) of the RMs.
table['catalog']='2008ApJ...688.1029M'  #This is the ADS Bibcode for Van Eck et al. 2011

#Step 5: Save the RMTable. Available formats are FITS and tsv. CSV is not allowed
#   in case any string contain commas.
table.write_tsv('example_table.tsv')










