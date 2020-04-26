#!/usr/bin/env python

'''
AMiGA library of functions for processing and detailing meta-data.
'''

__author__ = "Firas Said Midani"
__version__ = "0.2.0"
__email__ = "midani@bcm.edu"

# TABLE OF CONTENTS (15 functions)

# assembleMappings
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

import os 
import pandas as pd

from string import ascii_uppercase
from tabulate import tabulate

from libs import biolog
from libs.comm import smartPrint
from libs.org import assemblePath

def assembleMappings(data,mapping_path,meta_path,verbose):
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

        # see if user provided a mapping file that corresponds to this data file (filebase)
        if os.path.exists(mapping_file_path):

            df_mapping = pd.read_csv(mapping_file_path,sep='\t',header=0,index_col=0)   
            df_mapping = checkPlateIdColumn(df_mapping,filebase) # makes sure Plate_ID is a column

            smartPrint('{:.<30} Reading {}.'.format(filebase,mapping_file_path),verbose)

        # see if user described the file in meta.txt 
        elif filebase in meta_df_plates:

            meta_info = meta_df[meta_df.Plate_ID==filebase]
            msg = '{:.<30} Found meta-data in meta.txt '.format(filebase)

            biolog = isBiologFromMeta(meta_info)  # does meta_df indicate this is a BIOLOG plate

            if biolog:
                df_mapping = expandBiologMetaData(meta_info)
                msg += '& seems to be a BIOLOG PM plate.'
                smartPrint(msg,verbose)
            else:
                df_mapping = initKeyFromMeta(meta_info,well_ids)
                msg += '& does not seem to be a BIOLOG PM plate.'
                smartPrint(msg,verbose)

        elif isBiologFromName(filebase):

            df_mapping = initBiologPlateKey(filebase)
            msg = '{:.<30} Did not find mapping file or meta-data '.format(filebase)
            msg += 'BUT seems to be a BIOLOG PM plate.'
            smartPrint(msg,verbose)

        else:
            df_mapping = initMappingDf(filebase,well_ids) 
            msg = '{:.<30} Did not find mapping file or meta-data '.format(filebase)
            msg += '& does not seem to be a BIOLOG PM plate.'
            smartPrint(msg,verbose)

        df_mapping_dict[filebase] = expandMappingParams(df_mapping)

    #df_mapping = df_mapping.reset_index(drop=False)
    smartPrint('',verbose)

    return df_mapping_dict


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

    exists = os.path.exists(filepath)

    if not exists:
        df_meta = pd.DataFrame
        df_meta_plates = [];
    else:
        df_meta = pd.read_csv(filepath,sep='\t',header=0,index_col=None)

    # which plates were characterized in meta.txt?
    try:
        df_meta_plates = df_meta.Plate_ID.values
    except:
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

    if '_' in filebase:
        split_filebase = filebase.split('_')
    else:
        return False

    if not split_filebase[1].startswith('PM'):
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

    isolate = str(plate_id.split('_')[0])
    pmn = int(plate_id.split('PM')[1][0])
    rep = [int(plate_id.split('-')[-1]) if '-' in plate_id else 1][0]

    if simple:
        return isolate,pmns
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


def expandMappingParams(df):
    '''
    Expand input data frame to include columns relevant for AMiGA processing of user paramters.
        It will add columns for Group, Control, Flag, and Subset.

    Args:
        df (pandas.DataFrame)

    Returns:
        df (pandas.DataFrame): with four additional columns
    '''

    # plate can be divided into multiple group, each gorup has unique control well(s)
    df.loc[:,'Group'] = [1]*df.shape[0]  # all wells in a BIOLOG PM plte belong to same group
    df.loc[:,'Control'] = [0]*df.shape[0]  # all wells (except) A1 are treatments
    df.loc['A1','Control'] = 1  # A1 is the control well

    # initialize well-specific flag and subset parameters
    df.loc[:,'Flag'] = [0]*df.shape[0]  # by default, no wells are flagged
    df.loc[:,'Subset'] = [1]*df.shape[0]  # by default, all wells are included in analysis

    return df


def parseBiologLayout():
    '''
    Initializes a pandas.DataFrame that maps the location (well and plate number)
        of each sugar in BIOLOG PM plates.

    Retruns:
        df (pandas.DataFrame)
    '''

    biolog_layout = [biolog.Carbon1,biolog.Carbon2,biolog.PhosphorusAndSulfur]
    biolog_layout += [biolog.PeptideNitrogen1,biolog.PeptideNitrogen2,biolog.PeptideNitrogen3]

    index = [str(ii) for ii in range(1,7)]
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
                list_wells.append('{}{}'.format(row,col))
                list_rows.append(row)
                list_cols.append(col)
    else:
        for row in rows:
            for col in cols:
                list_wells.append('{}{}'.format(row,col))
                list_rows.append(row)
                list_cols.append(col)

    # assemble dataframe
    df = pd.DataFrame([list_wells,list_rows,list_cols],index=['Well','Row','Column'])
    df = df.T  # transpose to long format 
    df = df.set_index('Well')

    return df


