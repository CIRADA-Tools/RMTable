#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Example code demonstrating the use of the RMTable class.

Created on Fri Apr 26 09:42:09 2019
@author: cvaneck
"""

import numpy as np
from rmtable import RMTable
from astropy.coordinates import SkyCoord


#Reading in an RMTable from FITS:
catalog=RMTable.read('VanEck2011_table.fits')
print(catalog)

#Get the list of columns present in table:
print(catalog.columns)

#Get number of RMs (2 methods):
print(catalog.size)
print(len(catalog))

#Access column(s):
print(catalog['rm'])
# Note here that missing columns will always be added
print(catalog['l','b','rm','rm_err'])
# Instead you can covert back to a standard table
print(catalog.to_table()['l','b','rm','rm_err'])

#Access row(s):
print(catalog[0:10])

#Extracting a subset of the catalog:
selection=np.logical_and(catalog['l'] > 90,catalog['l'] < 270)
#This is an array of booleans, that can be used to extract a portion of the catalog.
print(catalog[selection])
#Multiple selections are combined using numpy's logical_and function.


#If you prefer numpy arrays or pandas dataframes, convert to those
print(catalog.as_array())
print(catalog.to_pandas())


#Saving a sub-catalog:
subcatalog=catalog[selection]
subcatalog.write('subcatalog.fits',overwrite=True)






#How to convert other tables containing RMs into RMTables.
#Read in machine-readable table (fixed-width ASCII file) of catalog,
#into numpy ndarray. Note the fixed width columns are set with the ’delimiter’ 
#keyword. Columns that match the standard are given names in the standard, 
#to allow direct conversion; other columns must avoid name conflicts with
#standard columns.
cat=np.genfromtxt('VanEck2011.dat',encoding=None,dtype=None,
                  delimiter=[6,6,3,3,5,2,2,3,3,5,4,5,3,5,3,7,5],
                    names=['l','b','rah','ram','ras','dec_sign','decd',
                           'decm','decs','stokesI','polint','rm','rm_err',
                           'RMSynth','dRMsynth','NVSSRM','dNVSSRM'])
#Setting the column names correctly is important to get the data into the RMTable.
# Columns with incorrect names are ignored in the conversion process.



#The RA and Dec columns must be converted from sexigessimal to decimal.
#The easiest way is to use Astropy's capability to read 'hms dms' strings:
ra_strings=np.char.add(np.char.add(np.char.add(cat['rah'].astype(str),'h'),
                                   np.char.add(cat['ram'].astype(str),'m')),
                                   np.char.add(cat['ras'].astype(str),'s'))
dec_strings=np.char.add(cat['dec_sign'],
                        np.char.add(np.char.add(np.char.add(cat['decd'].astype(str),'d'),
                                                np.char.add(cat['decm'].astype(str),'m')),
                                                np.char.add(cat['decs'].astype(str),'s')))
coords=SkyCoord(ra_strings,dec_strings,frame='fk5')
#Adding final decimal coordinate columns in to the numpy table:
cat=np.lib.recfunctions.append_fields(cat,['ra','dec'],[coords.ra.deg,coords.dec.deg])


#Step 2: do necessary unit conversions to match RMTable convention.
#In this example, converting fluxes from mJy in the input table to Jy in the RMTable.
cat['polint']=cat['polint']/1e3
cat['stokesI']=cat['stokesI']/1e3


#Step 3: convert to RMTable. It will automatically identify which columns are
#        part of the standard and which are not, based on the column names.
table = RMTable(cat)
#Tt will report which columns were used or ignored, and which
#are missing and filled with blanks.
#If verify=True, it will check that the numerical values are as expected.
#This typically means things like angle conventions (i.e. polarization angles 
#from [0,180)).
#The coordinate system must be specified to ensure that coordinates are 
#successfully converted to ICRS.


#Step 4: add any information that wasn't in the input table (but is in the text
#        of the paper. Most important is the catalog bibcode.
table['catalog']='2011ApJ...728...97V'
table['rm_method']='EVPA-linear fit'
table['ionosphere']='None'
table['flux_type']='Peak'
table['beam_maj']=0.01388888889
table['minfreq']=1365e6
table['maxfreq']=1515e6
table['channelwidth']=3.57e6
table['Nchan']=14
table['noise_chan']=2e-3
table['int_time']=120

#The following lines check the table values for conformance with the standard
# (within limits, and using standard string values where applicable).
# Note that these methods are always run automatically, but can be run manually
table.verify_limits()
table.verify_standard_strings()


#Step 5: Save the RMTable. All formats supported by astropy are supported.
table.write('VanEck2011_table.fits',overwrite=True)









