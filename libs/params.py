#!/usr/bin/env python

'''
AMiGA library for handling and reporting growth parameters.
'''

__author__ = "Firas S Midani"
__email__ = "midani@bcm.edu"


# TABLE OF CONTENTS (7 functions)

# initDiauxieList
# initParamList
# initParamDf
# mergeDiauxieDf
# minimizeParameterReport
# minimizeDiauxieReport
# prettifyParameterReport

import sys
import numpy as np
import pandas as pd

from scipy.stats import norm

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
               't_k','t_gr','t_dr','diauxie']
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


def prettyifyParameterReport(df,target,confidence=0.975):
    '''
    Called by compare.py to create a human-readable comparison of parameter estimates
        between two user-selected conditions. The generated table will have parameters 
        as index column, the first few rows will be meta-data, the cell values will be 
        either the mean or confidence interval of estimates. 

    Args:
        df (pandas.DataFrame): Two samples (rows) by many growth parameters (columns).
            The growth parameters can be summay (mean and standard deviation) estimates 
            or simple estimate (mean only).
        target (string)
        confidence (float): must be between 0.8 and 1.
    '''

    if df.shape[0] != 2:

        msg = 'FATAL USER ERROR: AMiGA can only contrast the confidence intervals of '
        msg += 'two experimental conditions. Please subest the input dataframe properly '
        msg += 'before submitting to prettifyParameterReport.'
        sys.exit(msg)


    def getConfInts(means,stds,conf=0.975):
        '''
        Computes confidence interval based on mean, standard deviation, and desired confidence.

        Args:
            means (array of floats)
            stds (array of floats)
            conf (float)

        Returns:
            cis (array of strings), where each formatted string indicates the confidence interval,
                e.g. [0.4,0.6]
        '''

        scaler = norm.ppf(conf)

        cis = []
        for m,s in zip(means,stds):
            low, upp = m-scaler*s, m+scaler*s
            cis.append('[{0:.3f},{1:.3f}]'.format(low,upp))

        return cis

    def detSigDiff(a,b):
        '''
        Detemines if there is a significant difference between two variables, 
            based on whether confidence intervals overlap or not

        Args:
            a (2-array): lower and upper bounds of first confidence interval
            b (2-array): lower and upper bounds of second confidence intervals

        Returns:
            (boolean): True (intervals do not overlap) or False (intervals overlap)
        '''

        a = [float(ii) for ii in a]
        b = [float(ii) for ii in b]
        return not ((a[0] <= b[1]) and (b[0] <= a[1]))


    confidence = 1-(1 - confidence)/2

    df = df.set_index(['Sample_ID'])

    params = list(set([ii.split('(')[1][:-1] if '(' in ii else ii for ii in df.T.index[1:]]))

    df_mus = pd.DataFrame(columns=df.iloc[:,0],index=params).sort_index()
    df_cis = pd.DataFrame(columns=list(df.iloc[:,0])+[''],index=params).sort_index()
    df_cis.T.index.name = target

    for p in params:

        if p == 'diauxie':
            df_mus.loc[p,:] = df.loc[:,p].values
            if df.loc[:,p].values[0] != df.loc[:,p].values[1]:
                df_cis.loc[p,:] = ['NA','NA',True]
            else:
                df_cis.loc[p,:] = ['NA','NA',False]
        else:
            mus = df.loc[:,'mean({})'.format(p)].values
            stds = df.loc[:,'std({})'.format(p)].values
            cis = getConfInts(mus,stds,confidence)
            olap = detSigDiff(eval(cis[0]),eval(cis[1]))

            df_mus.loc[p,:] = ['{0:.3f}'.format(ii) for ii in mus]
            df_cis.loc[p,:] = cis + [olap]

    df_mus.loc['Parameter',:] = ['Mean','Mean']
    df_cis.loc['Parameter',:] = ['{}% CI'.format(100*confidence),
                                 '{}% CI'.format(100*confidence),
                                 'Sig. Diff.']

    df_mus = df_mus.T.reset_index().T
    df_cis = df_cis.T.reset_index().T
    df_all = df_mus.join(df_cis,lsuffix='L',rsuffix='R')
    df_all = df_all.T.set_index([target,'Parameter']).T

    return df_all

def articulateParameters(df,axis=0):
    '''
    Replace shortnames for growth parameters with clear descriptions.

    Args:
        df (pandas.DataFrame)
        axis (0 or 1): whether parameters are index labels (0) or column names (1)

    Retuns:
        df (pandas.DataFrame)
    '''

    params_labels = {'auc_lin' : 'AUC (lin)',
                     'auc_log' : 'AUC (log)',
                     'death_lin' : 'Death (lin)',
                     'death_log' : 'Death (log)',
                     'diauxie' : 'Diauxie',
                     'dr' : 'Death Rate',
                     'gr' : 'Growth Rate',
                     'k_lin' : 'Carrying Capacity (lin)',
                     'k_log' : 'Carrying Capacity (log)',
                     'lagC' : 'Lag Time',
                     'lagP' : 'Adaptation Time',
                     'td' : 'Doubling Time',
                     't_dr' : 'Time at Max. Death Rate',
                     't_gr' : 'Time at Max. Growth Rate',
                     't_k' : 'Time at Carrying Capacity'}

    return df.rename(params_labels,axis=axis)


