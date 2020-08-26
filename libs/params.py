#!/usr/bin/env python

'''
AMiGA library for handling and reporting growth parameters.
'''

__author__ = "Firas S Midani"
__email__ = "midani@bcm.edu"


# TABLE OF CONTENTS (6 functions)

# initDiauxieList
# initParamList
# initParamDf
# mergeDiauxieDf
# minimizeParameterReport
# minimizeDiauxieReport

import numpy as np
import pandas as pd

from libs.utils import subsetDf, getValue


def initDiauxieList(params=None):
    '''
    Returns list of strings which are growth parameters specific to diauxie characterization.
    '''

    p = initParamList(0,params=params)
    p = ['dx_{}'.format(ii) for ii in p]
    p += ['dx_t0','dx_tf']
    p.remove('dx_diauxie')

    return p


def initParamList(complexity=0,params=None):
    '''
    Initialize a list of parameter variables. 

    Args:
        complexity (0,1,2): whether to include raw, mean and std, or norm version of parameters
        params(None or list): list of parameters to process

    Returns (list)
    '''


    if params is None:
        lp0 = ['auc_lin','auc_log','k_lin','k_log',
               'gr','dr','td','lagC','lagP','death_lin','death_log',
               'x_k','x_gr','x_dr','diauxie']
    else:
        lp0 = params

    lp1a = ['mean({})'.format(lp) for lp in lp0[:-1]]
    lp1b = ['std({})'.format(lp) for lp in lp0[:-1]]
    lp2 = ['norm({})'.format(ii) for ii in lp0[:-1]]

    if complexity == 0: return lp0
    elif complexity == 1: return lp1a + lp1b
    elif complexity == 2: return lp2


def initParamDf(sample_ids=None,complexity=0):
    '''
    Initialize lists of parameter names or empty pandas.DataFrame to store growth parameers.

    Args:
        sample_ids (list): used to index dataFrame rows.
        complexity (0,1,2): whether to include raw, mean and std, or norm version of parameters

    Return:
        (pandas.DataFame,pandas.DataFrame)
    '''

    params = initParamList(complexity=complexity)

    for ii in ['mean(diauxie)','std(diauxie)']:
        if ii in params: params.remove(ii)

    return pd.DataFrame(index=sample_ids,columns=params)


def mergeDiauxieDfs(diauxie_ls):
    '''
    Merges list of pandas.DataFrames into a single one vertically.

    Args (list)
    Returns (pandas.DataFrame)
    '''

    def addSampleID(df,sid):
        '''
        Adds Sample ID (str) as a column in a pandas.dataframe and returns the dataframe.
        '''
        df.index = [sid]*df.shape[0]
        df.index.name = 'Sample_ID'
        df = df.reset_index(drop=False)
        return df

    merged = {sid:addSampleID(df,sid) for sid,df in diauxie_ls.items()}
    merged = pd.concat(merged.values(),axis=0).reset_index(drop=True)

    return merged

def minimizeParameterReport(df):
    '''
    Minimizes a pandas.DataFrame to only inlcude parameters indicated in 
        the config.py file under 'report-parameters' variable. 

    Args (pandas.DataFrame)
    Return (pandas.DataFrame)
    '''

    request = getValue('report_parameters')
    request = initParamList(0,request) + initParamList(1,request) + initParamList(2,request)

    lp = initParamList(0) + initParamList(1) + initParamList(2)
    keys = set(lp).intersection(set(df.keys()))
    remove = keys.difference(set(request))

    return df.drop(remove,axis=1)


def minimizeDiauxieReport(df):
    '''
    Minimizes a pandas.DataFrame to only inlcude parameters indicated in 
        the config.py file under 'report-parameters' variable. 

    Args (pandas.DataFrame)
    Return (pandas.DataFrame)
    '''

    request = getValue('report_parameters')
    request = initDiauxieList(request)

    lp = initDiauxieList()
    keys = set(lp).intersection(set(df.keys()))
    remove = set(initDiauxieList()).difference(set(request))

    return df.drop(remove,axis=1)




