#!/usr/bin/env python

'''
DESCRIPTION library for auxiliary functions, primarily for data structure manipulations or data munging relevatn to AMiGA.
'''

__author__ = "Firas Said Midani"
__version__ = "0.2.0"
__email__ = "midani@bcm.edu"


# TABLE OF CONTENTS (7 functions)

# subsetDf
# resetNameIndex
# timeStamp
# getPlotColors
# getTextColors
# getValue
# getTimeUnits

import pandas as pd

from datetime import datetime

from libs.config import config

def subsetDf(df,criteria):
    '''
    Retains only rows in a pandas.DataFrame that match select criteria. 

    Args:
        df (pandas.DataFrame)
        criteria (dictionary): keys (str) are column headers in df, and values (list) are respective column values
    '''

    return df[df.isin(criteria).sum(1)==len(criteria)]


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

    if text=='input':
        return config['time_input_unit']
    elif text=='output':
        return config['time_output_unit']
