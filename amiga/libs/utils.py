#!/usr/bin/env python

'''
AMiGA library for auxiliary functions, primarily for data structure manipulations or data munging relevant to AMiGA.
'''

__author__ = "Firas Said Midani"
__email__ = "midani@bcm.edu"


# TABLE OF CONTENTS (16 functions)

# flattenList
# randomString
# uniqueRandomString
# subsetDf
# concatFileDfs
# raise_non_pos
# handle_non_pos
# resetNameIndex
# timeStamp
# selectFileName
# getPlotColors
# getTextColors
# getValue
# getTimeUnits
# getHypoPlotParams
# reverseDict

import os
import sys
import numpy as np # type: ignore
import pandas as pd # type: ignore
import string
import random

from datetime import datetime

from .config import config


def flattenList(ls): 
    '''Flatten a list of lists

    Args:
        ls (list): multi-dimensional

    Returns: 
        1-dimensional list
        '''

    return [item for sublist in ls for item in sublist]


def randomString(n=6):
    '''
    Generate a random string of size n.
    '''

    return ''.join([random.choice(string.ascii_letters) for n in range(n)])

def uniqueRandomString(n=6,avoid=list()):
    '''
    Generate a random string of size n (int) and make sure it does not conflict with avoid (list).
    '''

    rs = randomString(n=n)
    while rs in avoid:
        rs = randomString(n=n)
    return rs

def subsetDf(df,criteria):
    '''
    Retains only rows in a pandas.DataFrame that match select criteria. 

    Args:
        df (pandas.DataFrame)
        criteria (dictionary): keys (str) are column headers in df, and values (list) are respective column values

    Returns (pandas.DataFrame)
    '''

    if criteria is None:
        return df

    # if a criteria (dictionary) has a value that is not list format, put it into a list
    for key,value in criteria.items():
        if not isinstance(value,list):
            criteria[key] = [value]

    return df[df.isin(criteria).sum(axis=1,numeric_only=True)==len(criteria)]


def concatFileDfs(ls_files):
    '''
    Reads all files passed in sole argument into pandas.DataFrame objects then
        concatenates all into a single pandas.DataFrame

    Args:
        ls_files (list): list of file paths, files must be saved as tab-separated pandas.DataFrames

    Returns (pandas.DataFrame)
    '''

    ls_files = [lf for lf in ls_files if os.path.exists(lf)]

    if not ls_files:
        return pd.DataFrame() # if no files exist, return empty dataframe

    df = []
    
    for lf in ls_files:
        df.append(pd.read_csv(lf,sep='\t',header=0,index_col=0))

    return pd.concat(df,sort=False)


def raise_non_pos(arr):
    '''
    Replaces zero or negative values in array with lowest positive value found in array.

    Args:
        arr (list or np.array)

    Returns:
        arr (list)
    '''

    # find the lowest positive value (non-negative and non-zero)
    floor = min(i for i in arr if i > 0)
    
    # replace negative or zero values with floor
    arr = [floor if i <=0 else i for i in arr]
    
    return arr


def handle_non_pos(arr):
    '''
    Handles zero or negative values in an array.

    Args:
        arr (list or np.array)

    Returns:
        arr (list)
    '''

    # get parameters
    approach = config['handling_nonpositives']
    lod = config['limit_of_detection']
    force_lod = config['force_limit_of_detection']
    ndeltas = np.min([config['number_of_deltas'],len(arr)-1])
    delta_choice = config['choice_of_deltas']

    if approach == 'LOD' and lod == 0:
        msg = '\n\nFATAL USER ERROR: The limit-of-detection default value is set to 0. '
        msg+= 'It must be positive. You can adjust this parameter in the libs/config.py file.\n\n'
        sys.exit(msg)
    elif approach == 'Delta' and ndeltas < 1:
        msg = '\n\nFATAL USER ERROR: The number of deltas default value is less than 1 '
        msg+= 'It must be one or higher. You can adjust this parameter in the libs/config.py file.\n\n'

    # if the curve is simply a repeat of the same vlaue, you must use the LOD method
    if len(set(arr.values)) == 1:
        approach = 'LOD'

    # find the lowest value
    floor = min(arr)

    if approach == 'LOD':        

        # if all values are positive and not forcing limit-of-detection
        if floor > 0 and not force_lod:
            return arr

        # if minimum value is zero simpy shift by LOD
        elif floor == 0: 
            return arr + lod

        # if minimum value is negative but its absolute value is smaller than LOD
        elif floor < 0:
            return arr + lod + abs(floor)

        # if minimum value is poistive but user wants to force LOD 
        elif floor > 0 and force_lod and floor < lod:
            return arr + lod - floor

        else:
            return arr
    
    elif approach == 'Delta':

        if floor > 0:
            return arr

        delta = 0
        count = 2
        usr_ndeltas = ndeltas
        max_ndeltas = len(arr)-1

        while (delta == 0 or np.isnan(delta)):

            # compute absolute values of deltas over desired range
            deltas = [abs(arr[n]-arr[n-1]) for n in range(1,1+ndeltas)]

            # only consider positive deltas
            deltas = [ii for ii in deltas if ii>0]

            # pick delta based on desired operation
            if delta_choice == 'median':
                delta = np.median(deltas)
            elif delta_choice == 'mean':
                delta = np.mean(deltas)
            elif delta_choice == 'min':
                delta = np.min(deltas)
            elif delta_choice == 'max':
                delta = np.max(deltas)

            # increase ndeltas by a multiplier
            ndeltas = ndeltas * count
            count += 1

            # if next iteration won't yield new information, welp!
            if (delta == 0 or np.isnan(delta)) and (max_ndeltas - ndeltas) > usr_ndeltas: 
                return arr 

        if floor == 0:
            return arr + delta 

        elif floor < 0:
            return arr + delta + abs(floor)  # we have to add floor, o.w. minimum becomes zero

        else:
            return arr

def resetNameIndex(df,index_name='',new_index=False):
    '''
    Resets and names index of a pandas.DataFrame.

    Args:
        df (pandas.DataFrame): index (row.names), for mapping dataframes, index should be Well IDs (e.g. A1).
        index_name (str): name of index column, to be assigned.
        new_index (boolean): create new index (row number) and drop current, otherwise keep index as column

    Returns:
        mapping_df (pandas.DataFrame): with an additional coulmn with the header 'Well'.
    '''

    if not new_index: 
        df.index.name = index_name
        df.reset_index(drop=False,inplace=True)

    if new_index:
        df.reset_index(drop=True,inplace=True)
        df.index.name = index_name

    return df


def timeStamp():
    '''
    Reports the current time in a certain text format.

    Returns:
        ts (str)
    '''

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    return ts


def selectFileName(output):
    '''
    If user provided an output argument, return it, otherwise, return a time-stamp.

    Args:
        output (str): might be empty
    '''

    if output:
        return  output
    else:
        return timeStamp()  # time stamp, for naming unique files related to this operation


def getPlotColors(fold_change):
    '''
    Pulls from config (dictionary) desired value based on argument (str).

    Args:
        fold_change (str): limited set of options accepted

    Returns:
        (4-tuple) that indicates R,G,B values and lastly alpha (transparency) for plot object
        (4-tuple) that indicates R,G,B values and lastly alpha (transparency) for fill_between object
    '''

    if fold_change > config['fcg']:
        color_l = config['fcg_line_color']
        color_f = config['fcg_face_color']
    elif fold_change < config['fcd']:
        color_l = config['fcd_line_color']
        color_f = config['fcd_face_color']
    else:
        color_l = config['fcn_line_color']
        color_f = config['fcn_face_color']

    return color_l,color_f


def getTextColors(text):
    '''
    Pulls from config (dictionary) desired value based on argument (str).

    Args:
        text (str): limited set of options accepted

    Returns:
        (4-tuple) that indicates R,G,B values and lastly alpha (transparency) for text object
    '''

    if text=='OD_Max':
        return config['fcn_od_max_color']
    elif text=='Well_ID':
        return config['fcn_well_id_color']


def getValue(text):
    '''
    Pulls from config (dictionary) desired value based on argument (str).

    Args:
        text (str): limited set of options accepted 
    '''

    return config[text]


def getTimeUnits(text):
    '''
    Determines the time units desired for input and output based on settings in config.py

    Args (str): must be either 'input' or 'output'

    Returns (str) must be 'seconds', 'minutes', or 'hours'
    '''

    if text=='input':
        return config['time_input_unit']
    elif text=='output':
        return config['time_output_unit']

def getHypoPlotParams():

    return config['HypoPlotParams']

def reverseDict(foo):
    return {v: k for k, v in foo.items()}


