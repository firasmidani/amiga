#!/usr/bin/env python

'''
Script for parsing sample data from Dunphy (Cell Metabolism 2019) related to 
growth of mutants on N-Acetyl-Glucosamine and formatting them to be compatible with AMiGA. 
'''

__author__ = "Firas S Midani"
__email__ = "midani@bcm.edu"


import os
import pandas as pd
import numpy as np


if __name__ == '__main__':

    # pointers
    supplement='./non_cdiff/dunphy/supplement'
    working='./non_cdiff/dunphy/working'

    # create amiga working folder and sub-folders, if needed
    if not os.path.exists(working):  os.makedirs(working)
    if not os.path.exists('{}/data'.format(working)): os.makedirs('{}/data'.format(working))
    if not os.path.exists('{}/mapping'.format(working)): os.makedirs('{}/mapping'.format(working))

    # read data
    df1 = pd.read_csv('{}/S1A_Code/growthDataNAG_3.csv'.format(supplement),header=0)
    df2 = pd.read_csv('{}/S1A_Code/growthDataNAG_4.csv'.format(supplement),header=0)
    df = pd.concat([df1,df2])
    
    # subset data
    df = df[(df.expLabel!='blank') & (df.expLabel != 'blank_0')]
    df = df.drop(['replicate'],axis=1)
    df = df[df.media == 'N-Acetyl-D-Glucosamine']

    # define key (and map strain shortname to longname)
    key = pd.read_csv('{}/S1A_Code/NAG_mutants_key.csv'.format(supplement),header=0)
    key = key.set_index('expLabel').to_dict()['strain']
    key['Day0Anc'] = 'Ancestor'
    df = df.replace(key)

    # get median values for all conditions x timepoints
    df = df.groupby(['wellID','expLabel','timeTidy']).median().reset_index()

    # create AMiGA-friendly mapping file
    mapping = pd.DataFrame(df.loc[:,['wellID','expLabel']].values,columns=['Line','Strain'])
    mapping = mapping.drop_duplicates().reset_index(drop=True)
    mapping.loc[:,'Media'] = ['N-Acetyl-D-Glucosamine']*mapping.shape[0]
    mapping.loc[:,'Plate_ID'] = ['dunphy_glcnac_mutants']*mapping.shape[0]
    mapping.index = ['A'+str(ii) for ii in range(1,mapping.shape[0]+1)]
    mapping.index.name = 'Well_ID'    

    # create AMiGA-friendly data file with well IDs matching to mapping file well IDs
    line_to_wid = {v:k for k,v in mapping.loc[:,'Line'].to_dict().items()}
    df = df.replace(line_to_wid)
    df = df.pivot(index='timeTidy',columns='wellID',values='data').T
    
    # save files
    df.to_csv('{}/data/dunphy_glcnac_mutants.txt'.format(working),sep='\t',header=True,index=True)
    mapping.to_csv('{}/mapping/dunphy_glcnac_mutants.txt'.format(working),sep='\t',header=True,index=True)

