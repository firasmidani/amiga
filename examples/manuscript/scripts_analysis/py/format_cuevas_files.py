#!/usr/bin/env python

'''
Script for parsing sample data from PMAnalyzer and formatting them to be
compatible with AMiGA. 
'''

__author__ = "Firas S Midani"
__email__ = "midani@bcm.edu"


import os
import pandas as pd
import numpy as np


def format_data_file(filename):

    # read file
    fid = open(filename, 'r')
    lines = fid.read().strip('\n\n')
    lines = lines.split('\n\n')
    fid.close()

    # initialize pointers
    ls_ods = []

    # grab Well IDs, OD measurements, and time points
    for ii in lines:
        line = ii.split('\n')[6:96+6]
        ids = [ii.split('\t')[0] for ii in line]
        ods = [float(ii.split('\t')[-1]) for ii in line]
        ls_ods.append(ods)

    # assemble ODs into dataframe and label index and columns
    df = pd.DataFrame(ls_ods).T
    numTimePoints = df.shape[1]
    df.columns = np.linspace(0,1800*(numTimePoints-1),numTimePoints)
    df.index = ids

    return df

def format_mapping_file(filename):

    # These files are from 2009-2012 and it seems that they don't follow
    # the same layout as any of the current Biolog PM plates (circa 2020)
    df = pd.read_csv('./non_cdiff/cuevas/PMAnalyzer/plates/pm_plate_1.txt',
        sep='\t',header=None,index_col=None).iloc[:,:-1]
    df.columns = ['Well_ID','Source_Type','Substrate']
    df.loc[:,'Species'] = ['Citrobacter sedlakii']*df.shape[0]
    df.loc[:,'Plate_ID'] = [filename]*df.shape[0]

    group_dict = {'Carbon':1,'Nitrogen':2}
    df.loc[:,'Group'] = df.apply(lambda x: group_dict[x.Source_Type],axis=1)
    df.loc[:,'Control'] = df.apply(lambda x: [1 if x.Substrate=='Negative Control' else 0][0],axis=1)
    
    return df


if __name__ == '__main__':

    # pointers
    pmanalyzer='./non_cdiff/cuevas/PMAnalyzer'
    working='./non_cdiff/cuevas/working'

    # create amiga working folder and sub-folders, if needed
    if not os.path.exists(working):  os.makedirs(working)
    if not os.path.exists('{}/data'.format(working)): os.makedirs('{}/data'.format(working))
    if not os.path.exists('{}/mapping'.format(working)): os.makedirs('{}/mapping'.format(working))

    # these are the three sample fiels for C. sedlakii, create data and mapping files for each 
    for pid in ['R.S.3_ID773','R.S.3_ID952','R.S.3_ID953']:
        
        filename = '{}/sample/data_csedlakii/{}.txt'.format(pmanalyzer,pid)
        datafile = '{}/data/{}.txt'.format(working,pid)
        mappfile = '{}/mapping/{}.txt'.format(working,pid)

        data = format_data_file(filename)
        mapp = format_mapping_file(pid)

        data.to_csv(datafile,sep='\t',header=True,index=True)
        mapp.to_csv(mappfile,sep='\t',header=True,index=False)
