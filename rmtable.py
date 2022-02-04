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
        version='1.2'
        standard=[  
            ['ra','f8',[0,360],None],
            ['dec','f8',[-90,90],None],
            ['l','f8',[0,360],None],
            ['b','f8',[-90,90],None],
            ['pos_err','f4',[0,np.inf],np.nan],
            ['rm','f4',[-np.inf,np.inf],np.nan],
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
            ['pol_bias','U40','','Unknown'],
            ['flux_type','U40','','Unknown'],
            ['aperture','f4',[0,np.inf],np.nan],
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
            ['Nchan','i4',[0,np.inf],-2147483648],
            ['rmsf_fwhm','f4',[0,np.inf],np.nan],
            ['noise_chan','f4',[0,np.inf],np.nan],
            ['telescope','U80','','Unknown'],
            ['int_time','f4',[0,np.inf],np.nan],
            ['epoch','f4',[-np.inf,np.inf],np.nan],
            ['interval','f4',[0,np.inf],np.nan],
            ['leakage','f4',[0,np.inf],np.nan],
            ['beamdist','f4',[0,np.inf],np.nan],
            ['catalog','U40','',None],
            ['dataref','U400','',''],
            ['cat_id','U40','',''],
            ['type','U40','',''],
            ['notes','U200','','']        ]
        
        self.standard_columns=[x[0] for x in standard]
        self.standard_dtypes=[x[1] for x in standard]
        self.standard_limits=[x[2] for x in standard]
        self.standard_blanks=[x[3] for x in standard]
        
        
        self.table=at.Table(names=self.standard_columns,dtype=self.standard_dtypes)
        self.table.meta['VERSION']=version
        
        #These are for when extra columns might be added.
        #They point into the table where the columns can be found.
        self.columns=self.table.dtype.names
        self.dtype=[x[1] for x in self.table.dtype.descr]
        self.colcount=len(self.columns)
        
        self.size=0
    
    #Define standard entries for strings:
    standard_rm_method=['EVPA-linear fit','RM Synthesis - Pol. Int',
                        'RM Synthesis - Fractional polarization',
                        'RM Synthesis', 'QUfit', 'QUfit - Delta function',
                        'QUfit - Gaussian x Burn Slab',
                        'QUfit - Burn slab','QUfit - Gaussian','QUfit - Multiple',
                         'Unknown']
    standard_pol_bias=['1974ApJ...194..249W','1985A&A...142..100S','2012PASA...29..214G',
                       '1986ApJ...302..306K','Unknown','None','Not described']
    standard_telescope=['VLA','LOFAR','ATCA','DRAO-ST','MWA','WSRT','Effelsberg',
                        'ATA','ASKAP','Unknown']
    standard_classification=['','Pulsar','FRII hotspot','AGN','Radio galaxy','High-redshift radio galaxy','FRB']
    standard_flux_type=['Unknown','Integrated','Peak','Box','Visibilities','Gaussian fit - Peak']
    standard_complexity_test=['Unknown','None','Sigma_add','Second moment',
                              'QU-fitting']

    def __repr__(self):
        return self.table.__repr__()
    
    def _repr_html_(self):
        return self.table._repr_html_()

    def __str__(self):
        return self.table.__str__()
    
    def __len__(self):
        return len(self.table)

    def update_details(self):
        """Updates the 'actual' properties, which describe the columns present
        in the table, even if different from the standard.
        """
        self.columns=self.table.dtype.names
        self.dtype=[x[1] for x in self.table.dtype.descr]
        self.colcount=len(self.columns)
        self.size=len(self.table)
    
    def copy(self):
        """Creates a copy of the table as a new variable, which can be safely
        modified without affecting the original.
        """
        newtable=RMTable()
        newtable.table=self.table.copy()
        newtable.update_details()
        return newtable
    
    def write_FITS(self,filename,overwrite=False):
        """Write RMtable to FITS table format. Takes filename as input parameter.
        Can optionally supply boolean overwrite parameter."""
        self.table.write(filename,overwrite=overwrite)

    def read_FITS(self,filename):
        """Read in a FITS RMtable to this RMtable. Takes filename as input parameter. Overwrites current table entries.
        Does not confirm adherence to the standard: reads in as-is."""
        self.table=at.Table.read(filename)
        self.update_details()
        
    def write_tsv(self,filename):
        """Write RMtable to ASCII tsv table format. Takes filename as input parameter. Should automatically overwrite an existing file (not tested)
        Currently does not save non-standard columns! TODO: Fix this!"""
        #Check if string columns are internally represented as bytes or strings.
        #This causes no end of trouble since the replace functions need to have
        # maching types, and astropy.Table isn't fussy enough to be consistent.
        #Check for tabs in the string columns. If found convert to '@@' (does this have the same length?)        
        for i in range(len(self.columns)):
            if ('U' not in self.dtype[i]) and ('S' not in self.dtype[i]):
                continue
            w=np.where(np.char.find(self.table[self.columns[i]].data.astype(str),'\t') != -1)[0]
            if w.size > 0:
                print('Illegal tabs detected! Replaced with "@@"')
                self.table[self.columns[i]][w]=np.char.replace(self.table[self.columns[i]][w].data.astype(str),'\t','@@')
        #Similarly, check for newlines, and replace with double space ('  ')
            w=np.where(np.char.find(self.table[self.columns[i]].data.astype(str),'\n') != -1)[0]
            if w.size > 0:
                print('Illegal newlines detected! Replaced with "  "')
                self.table[self.columns[i]][w]=np.char.replace(self.table[self.columns[i]][w].data.astype(str),'\n','  ')

        
        row_string='{:}\t'*self.colcount
        row_string=row_string[:-1]+'\n' #Change last tab into newline.
        header_string='#'+'\t'.join(self.columns)+'\n'
        with open(filename,'w') as f:
            f.write(header_string)
            for i in range(len(self.table)):
                f.write(row_string.format(*self.table[i]))
    
    def read_tsv(self,filename):
        """Read in a ASCII tsv RMtable to this RMtable object. Takes filename as input parameter. 
        Overwrites existing table if present.
        Input: filename (str): path to .tsv file.
        """
        #Read in table (generically)
        table=np.genfromtxt(filename,dtype=None,names=True,delimiter='\t',encoding=None)
        
        #Try to get all the default columns into the correct formats.
        names=table.dtype.names
        dtypes=[]
        #If a defaut column, force to default type (to enforce typing/string length limits)
        #otherwise, use whatever it got read in as.
        for i in range(len(names)):         
            if names[i] in self.standard_columns:
                index=self.columns.index(names[i])
                dtypes.append(self.standard_dtypes[index])
            else:
                dtypes.append(table.dtype.fields[names[i]][0].str)


        self.table=at.Table(data=table,names=names,dtype=dtypes)
        self.update_details()
        
    def to_numpy(self):
        """Converts RMtable to Numpy ndarray (with named columns).
        Returns array."""
        return self.table.as_array()
    
    def input_numpy(self,array,verbose=False,verify=True,keep_cols=[],
                    coordinate_system='icrs'):
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
        
        Nrows=array.size

        #This function needs to:
        #Check that columns in ndarray have correct names, and no extra columns are present. (case senstitive?)
        #Check for missing essential columns?
        #Build new table, mapping correct input columns and adding appropriate blanks.
        #Generate second coordinate system if missing. 
        
        #Check for missing keep_col entries, and stop if they're not present in the array.
        missing_keep_cols=[ column for column in keep_cols if column not in array.dtype.names ]
        if len(missing_keep_cols) > 0:
            print('The following columns cannot be kept. Check spelling!')
            print(*missing_keep_cols,sep=',')
            raise RuntimeError()

        newtable=[None] * len(self.columns)   
        array_columns=array.dtype.names
        matching_columns=[]  #Track 'good' columns in input array
        additional_columns=[]  #Track 'useless' columns in input array
        missing_columns=list(self.columns).copy()  #Track columns not present in table
        for col in array_columns:
            if col in self.columns:  #Put the matching columns in the correct places
                matching_columns.append(col)
                missing_columns.remove(col)
                newtable[self.columns.index(col)]=array[col]
            elif col in keep_cols:
                pass
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
        #Check frame and convert to ICRS if needed, overwriting the ra and dec columns:
        if ('ra' in matching_columns) and (coordinate_system is not 'icrs'):
            print(f'Converting coordinates from {coordinate_system}.\n')
            coords=ac.SkyCoord(array['ra'],array['dec'],frame=coordinate_system,
                               unit='deg')
            array['ra']=coords.icrs.ra.deg
            array['dec']=coords.icrs.dec.deg
        
        
        #Check position columns, generate missing ones
        if ('ra' in matching_columns) and ('l' in matching_columns):
            pass  #Both present, all is well.
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
            raise RuntimeError()
        
        #Create missing columns with blank values:
        print("Missing columns (filling with blanks):")
        for col in missing_columns:
            newtable[self.columns.index(col)]=np.repeat(self.standard_blanks[self.columns.index(col)],Nrows)
            print(col)
        print()

        for i in range(len(newtable)):
            newtable[i]=at.Column(data=newtable[i],name=self.columns[i],dtype=self.standard_dtypes[i])

        #Turn everything into a RMtable, and append:
        newrows=at.Table(data=newtable,names=self.columns,dtype=self.standard_dtypes)
        self.table=at.vstack([self.table,newrows],join_type='exact')
        
        #Add kept columns:
        for column in keep_cols:
            self.add_column(array[column],column)
    
        if verify==True:
            self.verify_limits()
        self.update_details()
        return self

    
    def to_pandas(self):
        """Converts RMtable to pandas dataframe. Returns dataframe."""
        return self.table.to_pandas()
    
    def __getitem__(self,key):
        #Returns row, column, or table objects, as determined by astropy.table.
        value=self.table[key]
        #If the returned object is a whole table, wrap it back inside an RMTable
        if type(value) == at.table.Table:  
            subtable=RMTable()
            subtable.table=value
            subtable.update_details()
            value=subtable
        return value

    def __setitem__(self,key,item):
        self.table[key]=item
        
    def verify_columns(self):
        if self.standard_columns != self.table.colnames:
            print("Columns inconsistent with standard. Check if deliberate (i.e., verify these aren't misspellings).")
            extra_columns=[ column for column in self.columns if column not in self.standard_columns ]
            missing_columns=[ column for column in self.standard_columns if column not in self.columns ]
            print('Extra columns: ',end='')
            print(extra_columns)
            print('Missing colunns: ',end='')
            print(missing_columns)
            #print(set(self.table.colnames).symmetric_difference(set(self.standard_columns)))
    

    def verify_limits(self):
        """This function checks that all numerical columns conform to the standard for limits 
        on allowed numerical values. Mostly important for angles, as the standard uses [0,180) and not (-90,90].
        Non-conforming entries should be checked and fixed before incorporation into the master catalog."""
        good=True #Remains true until a non-conforming entry is found.
        for i in range(len(self.standard_columns)):
            if self.standard_columns[i] not in self.columns: #ignore missing columns.
                continue
            if self.standard_limits[i]=='': #ignore string columns
                continue
            data=self.table[self.standard_columns[i]]
            overmax=(data > self.standard_limits[i][1]).sum() #count how many outside of acceptable range
            undermin=(data < self.standard_limits[i][0]).sum()
            if overmax+undermin > 0:
                print('Column \'{}\' has {} entries outside the range of allowed values!'.format(self.standard_columns[i],overmax+undermin))
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
            
        invalid_complexity_test=[]
        for entry in self.table['flux_type']:
            if (entry not in self.standard_complexity_test) and (entry not in invalid_complexity_test):
                invalid_complexity_test.append(entry)
        if len(invalid_complexity_test) > 0:
            print('The following non-standard complexity test type(s) were found (at least once each):')
            print(*invalid_complexity_test,sep='\n')
        

        if len(invalid_methods+invalid_polbias+invalid_telescope+invalid_type+invalid_flux) == 0:
            print('No problems found with standardized string entries.')
            
    def append_to_table(self,table2,join_type='exact'):
        """This function concatenates a second RMtable to the end of this one.
        Input parameter: table2, an RMtable object.
                        join_type (str): 'exact' - requires both tables have exactly the same columns.
                                        'inner' - only columns common to both.
                                        'outer' - keep all columns, masking any missing from one table.
        Output: None"""
        self.table=at.vstack([self.table,table2.table],join_type=join_type)
        self.size=len(self.table)
        self.update_details()
        
    def add_column(self,data,name):
        """Add a new column to a table.
        Inputs: data (list- or array-like): column data. 
                        (Also accepts scalars, which are repeated to all rows)
                name (str): name of the new column.
        """
        self.table.add_column(at.Column(data=data,name=name))
        self.update_details()
        
    def add_missing_columns(self):
        """Adds in any missing default columns, with their default (blank) values.
        Can be used to make a table compliant with the standard in terms of having
        all the columns.
        """
        missing_columns=[ column for column in self.standard_columns if column not in self.columns ]
        for column in missing_columns:
            i=self.standard_columns.index(column)
            if self.standard_blanks[i] == None:
                raise Exception('Missing essential column: {}'.format(column))
            self.table.add_column(at.Column(data=self.standard_blanks[i],
                                            name=column,dtype=self.standard_dtypes[i]))
        self.update_details()

        
        
    def remove_column(self,name):
        """Removes an existing column. Will not allow removal of a default column.
        Inputs: name (str): name of the column to be removed.
        """
        self.table.remove_column(name)
        self.update_details()




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
    cat=RMTable()
    cat.read_FITS(filename)
    return cat
read_FITS.__doc__=RMTable.read_FITS.__doc__


def read_tsv(filename):
    cat=RMTable()
    cat.read_tsv(filename)
    return cat
read_tsv.__doc__=RMTable.read_tsv.__doc__


def input_numpy(array,verbose=False,verify=False,keep_cols=[],coordinate_system='icrs'):
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

    cat=RMTable()
    cat.input_numpy(array,verbose=verbose,verify=verify,keep_cols=keep_cols,
                    coordinate_system=coordinate_system)
    return cat
input_numpy.__doc__=RMTable.input_numpy.__doc__


def from_table(table):
    """Converts an Astropy table (with the correct columns) into an RMTable.
    Useful when sub-selection of a larger RMTable has returned an astropy table.
    Returns RMTable."""
    cat=RMTable()
    cat.table=table
    cat.update_details()
    return cat

def merge_tables(table1, table2,join_type='exact'):
    """Merges two RMTables into a single table.
    Inputs: table1 (RMTable): first table to be merged.
            table2 (RMTable): second table to be merged.
            join_type (str): 'exact' - requires both tables have exactly the same columns.
                             'inner' - only columns common to both.
                             'outer' - keep all columns, masking any missing from one table.
    """
    new_table=table1.copy() #Copy old table to avoid affecting that variable.
    new_table.update_details()
    new_table.append_to_table(table2,join_type=join_type)
    return new_table


def convert_angles(angles):
    """Converts an array of angles to follow the [0,180) degree convention
    used in the RMTable standard.
    Inputs: angles (array-like): angles (in degrees)
    Returns: array of angles (in degrees) in the range [0,180)
    """
    #Multiple of 180° that should be added to force all values to be positive
    n=np.ceil(np.abs(np.min(angles))/180)
    
    return np.mod(angles+180*n,180)
    



