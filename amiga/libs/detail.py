#!/usr/bin/env python

'''
AMiGA library of functions for processing and detailing meta-data.
'''

__author__ = "Firas Said Midani"
__email__ = "midani@bcm.edu"


# TABLE OF CONTENTS (18 functions)

# assembleMappings
# checkBiologSize
# checkMetaText
# checkPlateIdColumn
# isBiologFromMeta
# grabFirstValueFromDf
# expandBiologMetaData
# initSubstrateDf
# isBiologFromName
# initKeyFromMeta
# initBiologPlateKey
# parsePlateName
# initMappingDf
# expandMappingParams
# parseBiologLayout
# parseWellLayout
# updateMappingControls
# shouldYouSubtractControl

import os 
import sys
import numpy as np
import pandas as pd

from string import ascii_uppercase
from tabulate import tabulate

from . import biolog
from .comm import smartPrint
from .org import assembleFullName, assemblePath
from .utils import subsetDf


def assembleMappings(data,mapping_path,meta_path=None,save=False,verbose=False):
    '''
    Creates a master mapping file (or dictionary ?) for all data files in the input argument.
        For each data file, in this particular order, it will first (1) check if an individual
        mapping file exists, (2) if not, check if relevant meta-data is provided in meta.txt
        file, (3) if not, infer if plate is a BIOLOG PM based on its file name, and (4) if all
        fail, create a minimalist mapping file. 

    Args:
        data (dictionary): keys are file names (i.e. filebases or Plate IDs) and values are
            pandas DataFrames where index column (row names) are well IDs.
        mapping_path (str): path to the mapping folder.
        meta_path (str): path to the mapping file.
        verbose (boolean)

    Returns:
        df_mapping_dict (dictionary): keys are file names and values are mapping files. 
    '''

    df_mapping_dict = {}

    # list all data files to be analyed
    list_filebases = data.keys()

    # list all potential mapping file paths
    list_mapping_files = [assemblePath(mapping_path,ii,'.txt') for ii in list_filebases]

    # read meta.txt and list all plates described by it
    meta_df, meta_df_plates = checkMetaText(meta_path,verbose=verbose)   

    # assemble mapping for one data file at a time
    for filebase,mapping_file_path in zip(list_filebases,list_mapping_files):

        # what are the row names from the original data file 
        well_ids = data[filebase].columns[1:]  # this may no be A1 ... H12, but most ofen will be

        # create file path for saving derived mapping, if requested
        newfilepath = assembleFullName(mapping_path,'',filebase,'','.map')

        # see if user provided a mapping file that corresponds to this data file (filebase)
        if os.path.exists(mapping_file_path):

            df_mapping = pd.read_csv(mapping_file_path,sep='\t',header=0,index_col=0, dtype={'Plate_ID':str,'Isolate':str})
            df_mapping = checkPlateIdColumn(df_mapping,filebase) # makes sure Plate_ID is a column
            df_mapping.index = [ii[0] + ii[1:].lstrip('0') for ii in df_mapping.index] # strip leading zeros in well names

            smartPrint(f'{filebase:.<30} Reading {mapping_file_path}.',verbose=verbose)
        
        # see if user described the file in meta.txt 
        elif filebase in meta_df_plates:

            meta_info = meta_df[meta_df.Plate_ID==filebase]
            msg = f'{filebase:.<30} Found meta-data in meta.txt '

            biolog = isBiologFromMeta(meta_info)  # does meta_df indicate this is a BIOLOG plate

            if biolog:
                checkBiologSize(data[filebase],filebase)
                df_mapping = expandBiologMetaData(meta_info)
                msg += '& seems to be a BIOLOG PM plate.'
                smartPrint(msg,verbose=verbose)
            else:
                df_mapping = initKeyFromMeta(meta_info,well_ids)
                msg += '& does not seem to be a BIOLOG PM plate.'
                smartPrint(msg,verbose=verbose)

        elif isBiologFromName(filebase):
            checkBiologSize(data[filebase],filebase)
            df_mapping = initBiologPlateKey(filebase)
            msg = f'{filebase:.<30} Did not find mapping file or meta-data '
            msg += 'BUT seems to be a BIOLOG PM plate.'
            smartPrint(msg,verbose=verbose)

        else:
            df_mapping = initMappingDf(filebase,well_ids) 
            msg = f'{filebase:.<30} Did not find mapping file or meta-data '
            msg += '& does not seem to be a BIOLOG PM plate.'
            smartPrint(msg,verbose=verbose)

        df_mapping_dict[filebase] = expandMappingParams(df_mapping,verbose=verbose)

        if save:
            df_mapping_dict[filebase].to_csv(newfilepath,sep='\t',header=True,index=True)

    #df_mapping = df_mapping.reset_index(drop=False)
    smartPrint('',verbose=verbose)

    return df_mapping_dict


def checkBiologSize(df,filename):
    '''
    If a Plate_ID or meta.txt suggest a file corresponds to a BIOLOG PM, double check
        that it is at least a 96-well plate. 
    '''
    if df.shape[1] != 97: # (96 wells + 1 time) = 97 columns

        msg = f'The file "{filename}" is not formatted as a 96-well plate. '
        msg += 'Either the file name or its information in meta.txt '
        msg += 'suggsts that it is a BIOLOG PM plate. Please either correct the name '
        msg += 'of the file, the contents of the file, or correct its meta-data. '
        msg += 'If your file is missing Well IDs, you will need to add them in the first column. '
        msg += 'See documentation for more details.'

        sys.exit(msg)


def checkMetaText(filepath,verbose=False):
    '''
    Parses meta.txt file into a pandas.DataFrame.

    Args:
        filepath (str): path to meta.txt, must be tab-delimited with first column as "Plate_ID" (i.e. file name)
        verbose (boolean)

    Returns:
        df_meta (pandas.DataFrame)
        df_meta_plates (list)

    '''

    if filepath is None:
        exists = False
    else:
        exists = os.path.exists(filepath)

    if not exists:
        df_meta = pd.DataFrame
        df_meta_plates = []
    else:
        df_meta = pd.read_csv(filepath,sep='\t',header=0,index_col=None,dtype={'Plate_ID':str,'Isolate':str})

    # which plates were characterized in meta.txt?
    try:
        df_meta_plates = df_meta.Plate_ID.values
    except Exception:
        df_meta_plates = []

    # neatly prints meta.txt to terminal
    if exists:
        tab = tabulate(df_meta,headers='keys',tablefmt='psql')
        msg = '{:.<21}{}\n{}\n'.format('Meta-Data file is',filepath,tab)
        smartPrint(msg,verbose)
    else:
        smartPrint('No meta.txt file found\n',verbose)

    return df_meta,df_meta_plates


def checkPlateIdColumn(df,filebase):
    '''
    Validates that Plate ID is a column in dataframe.

    Args:
        df (pandas.DatamFrame)

    Returns:
        df (pandas.DataFrame) 
    '''

    if 'Plate_ID' not in df.keys():
        df.loc[:,'Plate_ID'] = [filebase]*df.shape[0]

    return df


def isBiologFromMeta(series):
    '''
    Identifies if a data file is a Biolog plate based on its meta-data.

    Args:
        ser (pandas.Series)

    Returns:
        (boolean)
    '''

    if 'PM' not in series.keys():
        return False
    elif grabFirstValueFromDf(series,'PM') in range(1,7): 
        return True
    else:
        return False


def grabFirstValueFromDf(df,key,fillna=None):
    '''
    Grabs the first value in a specific column in either a data frme or a series.

    Args:
        df (pandas.DataFrame or pandas.Series)
        key (str): column key
        fillna (str or None): if value not found, should replace with this argument

    Returns:
        value (data type is flexible)
    '''

    if key in df.keys():
        return df.loc[:,key].iloc[0]
    else:
        return fillna


def expandBiologMetaData(sr):
    '''
    Builds a pandas.DataFrame that combines file-specific meta-data with info derived from
        BIOLOG PM substrate lay-out. Datafrmae will include columns for Plate ID, Isolate 
        name, PM number, and Replicate number. 

    Args:
        sr (pandas.Series): describes meta-data for a specific file

    Returns:
        df (pandas.DataFrame)
    '''

    pmn = grabFirstValueFromDf(sr,'PM')  # PM plate number
    df_substrates = initSubstrateDf(pmn)  # mapping of wells to substrates
    df_meta = initKeyFromMeta(sr,df_substrates.index)
    df = df_meta.join(df_substrates)

    return df


def initSubstrateDf(pmn):
    '''
    Initializes a pandas DataFrame for substrates corresponding to a specific BIOLOG PM plate.

    Args:
        pmn (str or int or float): Phenotypic Microarray plate number, must be between [1,7)
    '''

    sr = parseBiologLayout().loc[:,str(int(pmn))]
    df = pd.DataFrame(sr)
    df.columns = ['Substrate']

    return df


def isBiologFromName(filebase):
    '''
    Determines if file name nomenclature indicates Biolog PM plate. Biolog PM nomenclature is
        {Isolate Name}_PM{PM plate number}-{Replicate number}.

    Args:
        filebase (str)

    Returns:
        (boolean) 
    '''

    if '_PM' in filebase:
        split_filebase = filebase.split('_PM')
    else:
        return False

    if split_filebase[1][0] not in [str(ii) for ii in range(1,7)]:
        return False

    return True


def initKeyFromMeta(series,well_ids):
    '''
    Initializes a mapping data frame using data derived from meta.txt. Here, it will reproduce
        a pandas.Series to match the number of wells in corresponding data file.

    Args:
        sr (pandas.Series): no size limit
        well_ids (list): list of variable data type

    Returns:
        df_meta (pandas.DataFrame): # rows = len(well_ids), # cols = df.shape[1]
    '''

    df_meta = pd.concat([series]*len(well_ids))
    df_meta.index = well_ids
    df_meta.index.name = 'Well'

    return df_meta


def initBiologPlateKey(plate_id,simple=False):
    '''
    Initiailzes a mapping dataframe using only Plate ID. Dataframe will have columns
        for Plate_ID, Isolate name, PM number, and Replicate number.

    Args:
        plate_id (str): data file name (i.e. filebase)
        simple (boolean): whether to include replicate number in mapping dataframe or not

    Returns:
        df_mapping (pandas.DataFrame)
    '''

    list_keys = ['Plate_ID','Isolate','PM','Replicate']
    isolate,pmn,rep = parsePlateName(plate_id,simple = False)
    df_meta = pd.DataFrame(index=list_keys,data=[plate_id,isolate,pmn,rep]).T
    df = expandBiologMetaData(df_meta)

    return df


def parsePlateName(plate_id,simple=False):
    '''
    Deconstructs a file name to identify Isolate name, PM plate number, and Replicate number.

    Args:
        plate_id (str)
        simple (boolean): whether to return Replicate number

    Returns:
        isolate (str)
        pmn (int)
        rep (int)
    '''

    plate_id_split = plate_id.split('_PM') 

    # in case isolate name includes "_PM" we will join all split strings except for last
    isolate = "_PM".join(plate_id_split[0:-1])

    # the number immediately fllowing the last instance of "_PM"
    pmn = int(plate_id_split[-1][0])
    
    # the number immediately following a hyphen after the last instance of "_PM", otherwise 1
    rep = [int(plate_id_split[-1].split('-')[-1]) if '-' in plate_id else 1][0]

    if simple:
        return isolate,pmn
    else:
        return isolate,pmn,rep


def initMappingDf(filebase,well_ids):
    '''
    Creates a minimalist file mapping (pandas.DataFrame). Index column are Well IDs, and sole
        value column is 'Plate_ID'. 

    Args:
        filebase (str): Plate ID
        well_ids (list of str): list of Well IDs (e.g. A1,A2,...,H11,H12)

    Returns:
        df_mapping (pandas.DataFrame): n x 1 where n is based on length of well_ids argument.
    '''

    data = [filebase]*len(well_ids)

    df_mapping = pd.DataFrame(index=well_ids,columns=['Plate_ID'],data=data)
    df_mapping.index.name = 'Well'

    return df_mapping


def expandMappingParams(df,verbose):
    '''
    Expand input data frame to include columns relevant for AMiGA processing of user paramters.
        It will add columns for Group, Control, Flag, and Subset. Note on grouping: plates
        can be divided into multiple groups where each group has its own group-specific 
        control wells. Biolog PM plates has a single group and A1 is control well. 

    Args:
        df (pandas.DataFrame)
        verbose (boolean)

    Returns:
        df (pandas.DataFrame): with four additional columns
    '''

    # get dataframe info
    Plate_ID = Plate_ID = df.Plate_ID.unique()[0]
    keys = list(df.keys()) 
    
    # check if Biolog PM plate
    biolog = isBiologFromName(Plate_ID) or isBiologFromMeta(df) # True or False

    if ('Control' in keys) and ('Group' not in keys):
        
        df.loc[:,'Control'] = df.Control.fillna(0)
        df.loc[:,'Group'] = [1]*df.shape[0]
                
    if ('Group' in keys) and ('Control' not in keys):

        df.loc[:,'Group'] = df.Group.fillna('NA')
        df.loc[:,'Control'] = [0]*df.shape[0]        
    
    if ('Group' not in keys) and ('Control' not in keys): 
    
        # plate can be divided into multiple group, each gorup has unique control well(s)
        df.loc[:,'Group'] = [1]*df.shape[0]  # all wells in a BIOLOG PM plte belong to same group
        df.loc[:,'Control'] = [0]*df.shape[0]  # all wells (except) A1 are treatments
    
    if biolog:      

        df.loc[:,'Control'] = 0  # A1 is the control well
        df.loc['A1','Control'] = 1  # A1 is the control well

    if not all(x in [0.,1.] or np.isnan(x) for x in df.Control.unique()):
        
        msg = '\nUSER ERROR: Values in Control column for mapping '
        msg += f'of {Plate_ID} must be either 0 or 1.\n'
        smartPrint(msg,verbose)

        df.loc[:,'Control'] = [0]*df.shape[0]

    # replace na values
    df.loc[:,'Group'] = df.Group.fillna('NA')
    df.loc[:,'Control'] = df.Control.fillna(0)

    # initialize well-specific flag and subset parameters
    if 'Flag' not in df.keys():
        df.loc[:,'Flag'] = [0]*df.shape[0]  # by default, no wells are flagged
    if 'Subset' not in df.keys():
        df.loc[:,'Subset'] = [1]*df.shape[0]  # by default, all wells are included in analysis

    return df


def parseBiologLayout():
    '''
    Initializes a pandas.DataFrame that maps the location (well and plate number)
        of each sugar in BIOLOG PM plates.

    Retruns:
        df (pandas.DataFrame)
    '''

    biolog_layout = [biolog.Carbon1,biolog.Carbon2,biolog.Nitrogen,biolog.PhosphorusAndSulfur]
    biolog_layout += [biolog.PeptideNitrogen1,biolog.PeptideNitrogen2,biolog.PeptideNitrogen3]

    index = [str(ii) for ii in range(1,8)]
    keys = parseWellLayout(order_axis=0).index

    biolog_layout_df = pd.DataFrame(biolog_layout,index=index,columns=keys).T
    biolog_layout_df.columns.name = 'PM'

    return biolog_layout_df 


def parseWellLayout(order_axis=1):
    '''
    Initializes a pandas.DataFrame where indices (row names) are well IDs (e.g. C8) and
        variables indicate row letter and column number.

    Args:
        order_axis (int): 0 indicates order by row (i.e. A1,A2,A3,...,B1,B2,B3,...) and
            1 indicates order by column (i.e. A1,B1,C1,...,A2,B2,C2,...).

    Returns:
        df (pandas.DataFrame, size 96x2): row names are well ID, and columns include
            row letter (str), and column number (str)
    '''

    # initialize rows = ['A',...,'H'] and cols = [1,...,12]
    rows = list(ascii_uppercase[0:8])
    cols = list(int(ii) for ii in range(1,13))

    list_wells = []
    list_cols = []
    list_rows = []

    # append one well at a time to list tht will later be merged into dataframe
    if order_axis == 1:
        for col in cols:
            for row in rows:
                list_wells.append(f'{row}{col}')
                list_rows.append(row)
                list_cols.append(col)
    else:
        for row in rows:
            for col in cols:
                list_wells.append(f'{row}{col}')
                list_rows.append(row)
                list_cols.append(col)

    # assemble dataframe
    df = pd.DataFrame([list_wells,list_rows,list_cols],index=['Well','Row','Column'])
    df = df.T  # transpose to long format 
    df = df.set_index('Well')

    return df


def updateMappingControls(master_mapping,mapping_dict,to_do=False):
    '''
    For all samples in master mapping, find relevant controls and add these controls to the master mapping dataframe.

    Args:
        master_mapping (pandas.DataFrame)
        mapping_dict (dictionary)
        to_do (boolean)

    Returns:
        master_mapping (pandas.DataFrame): will have more rows (i.e. samples) than input
    '''

    # check first if you need to do this
    if not to_do:
        return master_mapping

    # find all unique groups
    plate_groups = master_mapping.loc[:,['Plate_ID','Group']].drop_duplicates()
    plate_groups = [tuple(x) for x in plate_groups.values]

    # grab all relevant control samples
    df_controls = []
    for plate_group in plate_groups:
        pid,group = plate_group
        pid_mapping = mapping_dict[pid]
        df_controls.append(subsetDf(pid_mapping,{'Plate_ID':[pid],'Group':[group],'Control':[1]}))

    # re-assemble the master mapping dataframe, including the propercontrols
    df_controls = pd.concat(df_controls)
    master_mapping = pd.concat([master_mapping.copy(),df_controls.copy()],sort=True)
    master_mapping = master_mapping.reset_index(drop=True)
    master_mapping.index.name = 'Sample_ID'
    master_mapping = master_mapping.sort_values(['Plate_ID','Group','Control'])

    # if mapping has an interaction column, replace NaN with 0 (so it won't be dropped later)
    #   because you are (above) adding control samples to master_mapping,
    #   they will not have the interaction column and their values will default to NaN
    variable = [v for v in master_mapping.keys() if '*' in v]
    master_mapping.loc[:,variable] = master_mapping.loc[:,variable].fillna(0)

    return master_mapping


def shouldYouSubtractControl(mapping,variables):
    '''
    Checks if control samples must be subtracted from treatment samples for proper hypothesis testing.
        In particular, make sure that the variable of interest is binary (i.e. it has only two possible
        values in the mapping dataframe. This makes sure that GP regression on variable of interest is 
        performing a test on a binary variable.

    Args:
        mapping (pandas.DataFrame): samples (n) by variables (k)
        variable (str): must be one of the column headers for mapping argument

    Returns:
        (boolean)
    '''

    unique_values = mapping.loc[:,variables].drop_duplicates().reset_index()

    for _,row in unique_values.iterrows():
        criteria = row.to_dict()
        sub_map = subsetDf(mapping,criteria)
        sub_map_controls_n = sub_map[sub_map.Control==1].shape[0]
        sub_map_total_n = sub_map.shape[0]
        if (sub_map_controls_n == sub_map_total_n) and (sub_map_controls_n > 0):
            return False  # found a value whose samples all correspond to control wells
    else:
        return True
    