#!/usr/bin/env python

'''
AMiGA library of functions for trimming data and meta-data for AMiGA.
'''

__author__ = "Firas S Midani"
__email__ = "midani@bcm.edu"


# TABLE OF CONTENTS (6 functions)

# trimInput
# annotateMappings
# trimMergeMapping
# trimMergeData
# flagWells
# subsetWells

import copy
import pandas as pd # type: ignore
import sys

from functools import reduce

from .utils import subsetDf, resetNameIndex
from .comm import smartPrint, tidyDictPrint


def trimInput(data_dict,mapping_dict,params_dict=None,nskip=0,verbose=False):
    '''
    Interprets parameters to reduce mapping and
     data files to match user-desired criteria.

    Args:
        data (dictionary): keys are plate IDs and values are pandas.DataFrames with size t x (n+1)
            where t is the number of time-points and n is number of wells (i.e. samples),
            the additional 1 is due to the explicit 'Time' column, index is uninformative.
        mapping (dictionary): keys are plate IDs and values are pandas.DataFrames with size n x (p)
            where is the number of wells (or samples) in plate, and p are the number of variables or
            parameters described in dataframe.
        params (dictionary): must at least include 'subset' and 'flag' keys and their values
        verbose (boolean)

    Returns:
        data (dictionary): values may have smaller size than at time of input
        mapping (dictionary): values may have smaller size than at time of input 
    '''

    # a deep copy is needed to avoid altering mapping_dict globally
    mapping_dict = copy.deepcopy(mapping_dict)

    if params_dict is None:
        params_dict = {'subset': {}, 'flag': {}, 'hypothesis': {}, 'interval': {}}

    # annotate Subset and Flag columns in mapping files
    mapping_dict = annotateMappings(mapping_dict,params_dict,verbose)

    # trim and merge into single pandas.DataFrames
    master_mapping = trimMergeMapping(mapping_dict,verbose) # named index: Sample_ID
    master_data = trimMergeData(data_dict,master_mapping,nskip,verbose) # unnamed index: row number

    return master_data,master_mapping


def annotateMappings(mapping_dict,params_dict,verbose=False):
    '''
    Annotates the mapping data based on user-passed flags and subsetting criteria. In particular,
        it will annotate the Flag and Subset columns. It will also turn Well index into a
        standalone column. 

    Args:
        mapping_dict (dictionary): of mapping files (pandas.DataFrames), keys are file names (str)
        params_dict (dictionary): dictionary where keys are variables and values instances
        verbose (boolean)

    Returns:
        mapping_dict (dictionary): should be equal or smaller in size than input
    '''

    # flag wells that user does not want to analyze
    mapping_dict = flagWells(mapping_dict,params_dict['flag'],verbose=verbose,drop=False)

    # tag wells that meet user-passed criteria for analysis
    mapping_dict,_ = subsetWells(mapping_dict,params_dict['subset'],params_dict['hypothesis'],verbose=verbose)

    # make sure that mappings have Well columns
    #   here we assume that mapping_dict values have index of Well IDs, which should be the case
    mapping_dict = {pid:resetNameIndex(df,'Well',False) for pid,df in mapping_dict.items()}

    return mapping_dict


def trimMergeMapping(mapping_dict,verbose=False):
    '''
    Trims and merges mapping dataframes into one master mapping data frame.

    Args:
        mapping (dictionary): keys are plate IDs and values are pandas.DataFrames with size n x (p)
            where is the number of wells (or samples) in plate, and p are the number of variables or
            parameters described in dataframe.
        params (dictionary): must at least include 'subset' and 'flag' keys and their values
        verbose (boolean)

    Returns:
        mapping (pandas.DataFrame): number of wells/samples (n) x number of variables (p)  
    '''

    # merge mapping dataFrames
    #   sort will force shared (inner) keys to the lead and unshared (outer) keys to the caboose
    #   useful because individual mapping files may lack certain columns, some may even be empty 

    master_mapping = pd.concat(mapping_dict.values(),ignore_index=True,join='outer',sort=False)

    # trim mapping based on Subset and Flag columns
    master_mapping = subsetDf(master_mapping,{'Subset':[1],'Flag':[0,1]})

    # reset_index and set as Sample_ID
    master_mapping = resetNameIndex(master_mapping,'Sample_ID',True)

    return master_mapping


def trimMergeData(data_dict,master_mapping,nskip=0,verbose=False):
    '''
    Trims and merges data dataframes into one master dataframe. 

    Args:
        data (dictionary): keys are plate IDs and values are pandas.DataFrames with size t x (n+1)
            where t is the number of time-points and n is number of wells (i.e. samples),
            the additional 1 is due to the explicit 'Time' column, index is uninformative.
        mapping_df (pandas.DataFrame): number of well/samples (n) x number of variables (p)
        verbose (boolean)

    Returns:
        master_data (pandas.DataFrame): number of time points (t) x number of variables plus-one (p+1)
            plus-one because Time is not an index but rather a column
    '''

    new_data_dict = {}

    for pid,data in data_dict.items():

        # grab plate-specific samples
        mapping_df = master_mapping[master_mapping.Plate_ID==pid]
        
        # if no plate-specific samples are included in master_mapping, skip
        if mapping_df.shape[0]==0:
            continue

        # grab plate-specific data
        wells = list(mapping_df.Well.values)
        df = data.loc[:,['Time']+wells]

        # update plate-specific data with unique Sample Identifiers 
        sample_ids = list(mapping_df.index.values)
        df.columns = ['Time'] + sample_ids
        df.T.index.name = 'Sample_ID'

        # udpate dictionary
        df = df.iloc[nskip:,:]
        new_data_dict[pid] = df

    # each data is time (T) x wells (N), will merge all data and keep a single Time column
    #     reduce(fun,seq) applies a function (fun) recursively to all elements in list (seq)
    #     here, reduce will merge the first two dataFrames in data_dict.values(), then it
    #     will take this output and merge it with third dataFrame in list, and so on
    
    try: 
        master_data = reduce(lambda ll,rr: pd.merge(ll,rr,on='Time',how='outer'),new_data_dict.values())
    except TypeError:
        msg = "\nFATAL USER ERROR: AMiGA could not subset data based on user input. "
        msg += "Please check your arguments especially '-s' or '--subset' for any typos. "
        msg += "Keep in mind that AMiGA argument parser is case-sensitive.\n"
        sys.exit(msg)

    master_data = master_data.sort_values(['Time']).reset_index(drop=True)
    
    return master_data
    

def flagWells(df,flags,verbose=False,drop=False):
    '''
    Passes plate-well-specific flags from user into mapping dataframes.

    Args:
        df (dictionary of pandas.DataFrame) must have Plate_IDs and Well as columns
        flags (dictionary) with Plate_IDs (str) as keys and Wells (stR) as vlaues
        verbose (boolean)

    Returns:
        df (pandas.DataFrame)
    '''

    if (len(flags)==0):
        smartPrint('No wells were flagged.\n',verbose)
        return df

    for plate, wells in flags.items():
        if plate in df.keys():

            df[plate].loc[wells,'Flag'] = [1]*len(wells)

            if drop:
                df[plate] = df[plate][df[plate].Flag==0] 

    smartPrint('The following flags were detected:\n',verbose)
    smartPrint(tidyDictPrint(flags),verbose)

    return df


def subsetWells(df_mapping_dict,criteria,hypothesis,verbose=False):
    '''
    Tag wells that meet user-passed criteria.

    Args:
        df (pandas.DataFrame) must have Plate_IDs and Well as columns
        criteria (dictionary) with mapping variables (str) as keys and accepted instances (str) as values
        hypothesis (dictionary) 
        verbose (boolean)

    Returns:
        df (pandas.DataFrame)
    '''

    if (len(criteria)==0):
        smartPrint('No subsetting was requested.\n',verbose)
        return df_mapping_dict,None

    for plate_id,mapping_df in df_mapping_dict.items():

        # subsetting on number-based criteria does not match hits due to type mismatch (str vs int/float)
        mapping_df_str = mapping_df.astype(str)

        remove_boolean = ~(mapping_df_str.isin(criteria).sum(axis=1,numeric_only=True)==len(criteria)).values  # list of booleans
        remove_idx = mapping_df_str.index[remove_boolean]
        mapping_df.loc[remove_idx,'Subset'] = [0]*len(remove_idx)

    msg = 'The following criteria were used to subset data:\n'
    msg += tidyDictPrint(criteria)

    smartPrint(msg,verbose)

    return df_mapping_dict,msg

