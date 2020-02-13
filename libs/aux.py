#!/usr/bin/env python

'''
DESCRIPTION library for auxiliary functions, primarily for data structure manipulations or data munging relevatn to AMiGA.
'''

__author__ = "Firas Said Midani"
__version__ = "0.1.0"
__email__ = "midani@bcm.edu"


# TABLE OF CONTENTS

import pandas as pd

def subsetDf(df,criteria):
    '''
    Retains only rows in a pandas.DataFrame that match select criteria. 

    Args:
        df (pandas.DataFrame)
        criteria (dictionary): keys (str) are column headers in df, and values (list) are respective column values
    '''

    return df[df.isin(criteria).sum(1)==len(criteria)]

    
