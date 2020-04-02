#!/usr/bin/env python

'''
DESCRIPTION library for auxiliary functions, primarily for data structure manipulations or data munging relevatn to AMiGA.
'''

__author__ = "Firas Said Midani"
__version__ = "0.1.0"
__email__ = "midani@bcm.edu"


# TABLE OF CONTENTS

import pandas as pd

from libs.config import config

def subsetDf(df,criteria):
    '''
    Retains only rows in a pandas.DataFrame that match select criteria. 

    Args:
        df (pandas.DataFrame)
        criteria (dictionary): keys (str) are column headers in df, and values (list) are respective column values
    '''

    return df[df.isin(criteria).sum(1)==len(criteria)]

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

def getText(text):
    '''
    Pulls from config (dictionary) desired value based on argument (str).

    Args:
        text (str): limited set of options accepted

    Returns:
        (str) that is used for labels in plots. 
    '''

    if text=='grid_plot_y_label':
        return config['grid_plot_y_label']


def getTimeUnits(text):

    if text=='input':
        return config['time_input_unit']
    elif text=='output':
        return config['time_output_unit']

