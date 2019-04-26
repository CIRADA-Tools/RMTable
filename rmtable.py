import numpy as np
import astropy.table as at
import astropy.coordinates as ac
import astropy.units as au


"""
This module defines the RMTable class and all the methods necessary to read and write
RMTables into ASCII (tsv) or FITStable formats as well as convert new catalogs in the 
form of a numpy ndarray into new RMTables with the correct columns. Methods to convert
RMTables into numpy arrays or pandas dataframes are also included.
A full description of the RMTable standard is currently located at 
https://docs.google.com/document/d/1lo-W89G1X7xGoMOPHYS5japxJKPDamjEJ9uIGnRPnpo/edit
"""


class RMTable:
    """A class for holding tables of RMs and associated columns.
    Will have associated methods for reading, writing, outputting to various types """
    def __init__(self):
        # Column, dtype, [min,max],blank value
        version='1.0'
        standard=[  
            ['ra','f8',[0,360],None],
            ['dec','f8',[-90,90],None],
            ['l','f8',[0,360],None],
            ['b','f8',[-90,90],None],
            ['pos_err','f4',[0,np.inf],np.nan],
            ['rm','f4',[-np.inf,np.inf],None],
            ['rm_err','f4',[0,np.inf],np.nan],
            ['rm_width','f4',[0,np.inf],np.nan],
            ['rm_width_err','f4',[0,np.inf],np.nan],
            ['complex_flag','U1','','U'],
            ['complex_test','U80','',''],
            ['rm_method','U40','','Unknown'],
            ['ionosphere','U40','','Unknown'],
            ['Ncomp','i4',[1,np.inf],1],
            ['stokesI','f4',[0,np.inf],np.nan],
            ['stokesI_err','f4',[0,np.inf],np.nan],
            ['spectral_index','f4',[-np.inf,np.inf],np.nan],
            ['spectral_index_err','f4',[0,np.inf],np.nan],            
            ['reffreq_I','f4',[-np.inf,np.inf],np.nan],
            ['polint','f4',[0,np.inf],np.nan],
            ['polint_err','f4',[0,np.inf],np.nan],
            ['pol_bias','U50','','Unknown'],
            ['flux_type','U50','','Unknown'],
            ['fracpol','f4',[0,np.inf],np.nan],
            ['fracpol_err','f4',[0,np.inf],np.nan],
            ['polangle','f4',[0,180],np.nan],
            ['polangle_err','f4',[0,np.inf],np.nan],
            ['reffreq_pol','f4',[0,np.inf],np.nan],
            ['stokesQ','f4',[-np.inf,np.inf],np.nan],
            ['stokesQ_err','f4',[0,np.inf],np.nan],
            ['stokesU','f4',[-np.inf,np.inf],np.nan],
            ['stokesU_err','f4',[0,np.inf],np.nan],
            ['derot_polangle','f4',[0,180],np.nan],
            ['derot_polangle_err','f4',[0,np.inf],np.nan],
            ['stokesV','f4',[-np.inf,np.inf],np.nan],
            ['stokesV_err','f4',[0,np.inf],np.nan],
            ['beam_maj','f4',[0,np.inf],np.nan],
            ['beam_min','f4',[0,np.inf],np.nan],
            ['beam_pa','f4',[0,180],np.nan],
            ['reffreq_beam','f4',[0,np.inf],np.nan],
            ['minfreq','f4',[0,np.inf],np.nan],
            ['maxfreq','f4',[0,np.inf],np.nan],
            ['channelwidth','f4',[0,np.inf],np.nan],
            ['Nchan','i4',[0,np.inf],np.nan],
            ['noise_chan','f4',[0,np.inf],np.nan],
            ['telescope','U80','','Unknown'],
            ['int_time','f4',[0,np.inf],np.nan],
            ['epoch','f4',[-np.inf,np.inf],np.nan],
            ['interval','f4',[0,np.inf],np.nan],
            ['leakage','f4',[0,np.inf],np.nan],
            ['beamdist','f4',[0,np.inf],np.nan],
            ['catalog','U50','',''],
            ['dataref','U400','',''],
            ['cat_id','U40','',''],
            ['id','i4',[0,np.inf],np.nan],
            ['type','U40','',''],
            ['flagA_name','U40','',''],
            ['flagA_value','U40','',''],
            ['flagB_name','U40','',''],
            ['flagB_value','U40','',''],
            ['flagC_name','U40','',''],
            ['flagC_value','U40','',''],
            ['notes','U200','','']        ]
        
        self.columns=[x[0] for x in standard]
        self.dtypes=[x[1] for x in standard]
        self.limits=[x[2] for x in standard]
        self.blanks=[x[3] for x in standard]
        
        
        self.table=at.Table(names=self.columns,dtype=self.dtypes)
        self.table.meta['VERSION']=version
        
        self.size=0
    
    #Define standard entries for strings:
    standard_rm_method=['EVPA-linear fit','RM Synthesis - Pol. Int','RM Synthesis - Fractional polarization',
                         'QUfit - Delta function','QUfit - Burn slab','QUfit - Gaussian','QUfit - Complex','Unknown']
    standard_pol_bias=['1974ApJ...194..249W','1985A&A...142..100S','2012PASA...29..214G','Unknown']
    standard_telescope=['VLA','LOFAR','ATCA','DRAO-ST','MWA','Unknown']
    standard_classification=['','Pulsar','FRII hotspot','AGN','Radio galaxy','High-redshift radio galaxy','FRB']
    standard_flux_type=['Unknown','Integrated','Peak']

    def __repr__(self):
        return self.table.__repr__()

    def __str__(self):
        return self.table.__str__()
    
    def __len__(self):
        return len(self.table)

    
    def write_FITS(self,filename,overwrite=False):
        """Write RMtable to FITS table format. Takes filename as input parameter.
        Can optionally supply boolean overwrite parameter."""
        self.table.write(filename,overwrite=overwrite)

    def read_FITS(self,filename):
        """Read in a FITS RMtable to this RMtable. Takes filename as input parameter. Overwrites current table entries."""
        self.table=at.Table.read(filename)
        self.size=len(self.table)
        
    def write_tsv(self,filename):
        """Write RMtable to ASCII tsv table format. Takes filename as input parameter."""
        #Check for tabs in the string columns. If found convert to '@@' (does this have the same length?)
        for i in range(len(self.columns)):
            if 'U' not in self.dtypes[i]:
                continue
            w=np.where(np.char.find(self.table[self.columns[10]],'\t') != -1)[0]
            if w.size > 0:
                print('Illegal tabs detected! Replaced with "@@"')
                self.table[self.columns[10]][w]=np.char.replace(self.table[self.columns[10]][w],'\t','@@')
        
        row_string='{:}\t'*len(self.columns)
        row_string=row_string[:-1]+'\n' #Change last tab into newline.
        header_string='#'+'\t'.join(self.columns)+'\n'
        with open(filename,'w') as f:
            f.write(header_string)
            for i in range(len(self.table)):
                f.write(row_string.format(*self.table[i]))
    
    def read_tsv(self,filename):
        """Read in a ASCII tsv RMtable to this RMtable object. Takes filename as input parameter. Overwrites existing table."""
        table=np.genfromtxt(filename,dtype=self.dtypes,names=self.columns,delimiter='\t',encoding=None)
        self.table=at.Table(data=table,names=self.columns,dtype=self.dtypes)
        self.size=len(self.table)
        
    def to_numpy(self):
        """Converts RMtable to Numpy ndarray (with named columns).
        Returns array."""
        return self.table.as_array()
    
    def input_numpy(self,array,verbose=False,verify=True):
        """Converts an input numpy array into an RM table object.
        Requires that array has named columns matching standard column names.
        Will automatically fill in missing columns
        Input parameters: array (numpy ndarray): array to transform.
                          verbose (Boolean): report missing columns
                          verify (Boolean): check if values conform to standard."""
        
        Nrows=array.size
        self.size=Nrows

        #This function needs to:
        #Check that columns in ndarray have correct names, and no extra columns are present. (case senstitive?)
        #Check for missing essential columns?
        #Build new table, mapping correct input columns and adding appropriate blanks.
        #Generate second coordinate system if missing. 

        newtable=[None] * len(self.columns)   
        array_columns=array.dtype.names
        matching_columns=[]  #Track 'good' columns in input array
        additional_columns=[]  #Track 'useless' columns in input array
        missing_columns=self.columns.copy()  #Track columns not present in table
        for col in array_columns:
            if col in self.columns:  #Put the matching columns in the correct places
                matching_columns.append(col)
                missing_columns.remove(col)
                newtable[self.columns.index(col)]=array[col]
            else:
                additional_columns.append(col)
        
        #Report extraneous columns:
        if verbose:
            print('Incorporated columns:')
            print(*matching_columns,sep='\n')
            print()

            print('Unused columns (check for spelling/capitalization errors!):')
            print(*additional_columns,sep='\n')
            print()
        
        #Check position columns, generate missing ones
        if ('ra' in matching_columns) and ('l' in matching_columns):
            pass  #All is well.
        elif ('ra' in matching_columns) and ('l' not in matching_columns):
            #calculate Galactic:
            long,lat=calculate_missing_coordinates_column(array['ra'],array['dec'],True)
            newtable[self.columns.index('l')]=long
            newtable[self.columns.index('b')]=lat
            matching_columns.append('l')
            missing_columns.remove('l')
            matching_columns.append('b')
            missing_columns.remove('b')
        elif ('ra' not in matching_columns) and ('l' in matching_columns):
            #calculate Equatorial
            long,lat=calculate_missing_coordinates_column(array['l'],array['b'],False)
            newtable[self.columns.index('ra')]=long
            newtable[self.columns.index('dec')]=lat                          
            matching_columns.append('ra')
            missing_columns.remove('ra')
            matching_columns.append('dec')
            missing_columns.remove('dec')
        else:
            print('No position columns found in input table! Make sure RA/Dec or l/b columns haven\'t been lost!')
            raise RuntimeWarning()
        
        #Create missing columns with blank values:
        print("Missing columns (filling with blanks):")
        for col in missing_columns:
            newtable[self.columns.index(col)]=np.repeat(self.blanks[self.columns.index(col)],Nrows)
            print(col)
        print()

        for i in range(len(newtable)):
            newtable[i]=at.Column(data=newtable[i],name=self.columns[i],dtype=self.dtypes[i])

        #Turn everything into a RMtable, and append:
        newrows=at.Table(data=newtable,names=self.columns,dtype=self.dtypes)
        self.table=at.vstack([self.table,newrows],join_type='exact')
    
        if verify==True:
            self.verify_limits()
        return self
    
    def to_pandas(self):
        """Converts RMtable to pandas dataframe. Returns dataframe."""
        return self.table.to_pandas()
    
    def __getitem__(self,key):
        #Returns row, column, or table objects, as determined by astropy.table.
        return self.table[key]

    def __setitem__(self,key,item):
        self.table[key]=item
    

    def verify_limits(self):
        """This function checks that all numerical columns conform to the standard for limits 
        on allowed numerical values. Mostly important for angles, as the standard uses [0,180) and not (-90,90].
        Non-conforming entries should be checked and fixed before incorporation into the master catalog."""
        good=True #Remains true until a non-conforming entry is found.
        for i in range(len(self.columns)):
            if self.limits[i]=='': #ignore string columns
                continue
            data=self.table[self.columns[i]]
            overmax=(data >= self.limits[i][1]).sum() #count how many outside of acceptable range
            undermin=(data < self.limits[i][0]).sum()
            if overmax+undermin > 0:
                print('Column \'{}\' has {} entries outside the range of allowed values!'.format(self.columns[i],overmax+undermin))
                good=False
        if good == True:
            print('All columns conform with standard.')

    def verify_standard_strings(self):
        """This function checks the standardized string columns that they conform to the currently defined
        standard options. This is not strictly enforced, as the standard options are certainly incomplete.
        If assembling a catalog, please check that non-conforming values are not the result of typos,
        and contact the standard curator to have new options added to the standard."""
        invalid_methods=[]
        for entry in self.table['rm_method']:
            if (entry not in self.standard_rm_method) and (entry not in invalid_methods):
                invalid_methods.append(entry)
        if len(invalid_methods) > 0:
            print('The following non-standard RM method(s) were found (at least once each):')
            print(*invalid_methods,sep='\n')

        invalid_polbias=[]
        for entry in self.table['pol_bias']:
            if (entry not in self.standard_pol_bias) and (entry not in invalid_polbias):
                invalid_polbias.append(entry)
        if len(invalid_polbias) > 0:
            print('The following non-standard polarization bias correction method(s) were found (at least once each):')
            print(*invalid_polbias,sep='\n')

        invalid_telescope=[]
        for entry in self.table['telescope']:
            for scope in entry.split(','):
                if (scope not in self.standard_telescope) and (scope not in invalid_telescope):
                    invalid_telescope.append(scope)
        if len(invalid_telescope) > 0:
            print('The following non-standard telescope(s) were found (at least once each):')
            print(*invalid_telescope,sep='\n')

        invalid_type=[]
        for entry in self.table['type']:
            if (entry not in self.standard_classification) and (entry not in invalid_type):
                invalid_type.append(entry)
        if len(invalid_type) > 0:
            print('The following non-standard source classification(s) were found (at least once each):')
            print(*invalid_type,sep='\n')

        invalid_flux=[]
        for entry in self.table['flux_type']:
            if (entry not in self.standard_flux_type) and (entry not in invalid_flux):
                invalid_flux.append(entry)
        if len(invalid_flux) > 0:
            print('The following non-standard flux measurement type(s) were found (at least once each):')
            print(*invalid_flux,sep='\n')

        if len(invalid_methods+invalid_polbias+invalid_telescope+invalid_type+invalid_flux) == 0:
            print('No problems found with standardized string entries.')
            
#     standard_rm_method=['EVPA-linear fit','RM Synthesis - Pol. Int','RM Synthesis - Fractional polarization',
#                          'QUfit - Delta function','QUfit - Burn slab','QUfit - Gaussian','Unknown']
#     standard_pol_bias=['1974ApJ...194..249W','1985A&A...142..100S','2012PASA...29..214G','Unknown']
#     standard_telescope=['VLA','LOFAR','ATCA','DRAO-ST','MWA','Unknown']
#     standard_classification=['','Pulsar','FRII hotspot','AGN','Radio galaxy','High-redshift radio galaxy']
#     standard_flux_type=['Unknown','Integrated','Peak']

    def append_to_table(self,table2):
        """This function concatenates a second RMtable to the end of this one.
        Input parameter: table2, an RMtable object.
        Output: None"""
        self.table=at.vstack([self.table,table2.table],join_type='exact')
        self.size=len(self.table)

    
def calculate_missing_coordinates_column(long,lat,to_galactic):
    """Calculate a new pair of coordinate columns (equatorial/galactic) given the other pair and specified direction.
    Assumes input columns are already in degrees.
    Uses astropy coordinates for the transform.
    Input parameters:
        long: longitude column (ra/l)
        lat: latitude column (dec,b)
        to_galactic  (Boolean): direction of calculation: True = Equatorial -> Galactic, False = Galactic -> Equatorial
    Outputs: two arrays, new_long and new_lat
    """
    if to_galactic:
        sc=ac.SkyCoord(long,lat,frame='icrs',unit=(au.deg,au.deg))
        new_long=sc.galactic.l.deg
        new_lat=sc.galactic.b.deg
    else:
        sc=ac.SkyCoord(long,lat,frame='galactic',unit=(au.deg,au.deg))
        new_long=sc.icrs.ra.deg
        new_lat=sc.icrs.dec.deg

    return new_long,new_lat


def read_FITS(filename):
    """Read in a FITS RMtable to an RMtable object. Takes filename as input parameter. Returns RMTable."""
    cat=RMTable()
    cat.read_FITS(filename)
    return cat

def read_tsv(filename):
    """Read in a ASCII tsv RMtable to an RMtable object. Takes filename as input parameter. Returns RMTable."""
    cat=RMTable()
    cat.read_tsv(filename)
    return cat

def input_numpy(array,verbose=False,verify=False):
    """Converts an input numpy array into an RM table object.
    Requires that array has named columns matching standard column names.
    Will automatically fill in missing columns
    Input parameters: array (numpy ndarray): array to transform.
                      verbose (Boolean): report missing columns
                      verify (Boolean): check if values conform to standard.
    Returns RMTable."""
    cat=RMTable()
    cat.input_numpy(array,verbose=verbose,verify=verify)
    return cat

def from_table(table):
    """Converts an Astropy table (with the correct columns) into an RMTable.
    Useful when sub-selection of a larger RMTable has returned an astropy table.
    Returns RMTable."""
    cat=RMTable()
    cat.table=table
    cat.size=len(table)
    return cat
