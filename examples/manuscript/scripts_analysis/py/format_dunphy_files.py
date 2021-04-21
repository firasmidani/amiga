#!/usr/bin/env python

'''
Script for parsing sample data from Dunphy (Cell Metabolism 2019) and 
formatting them to be compatible with AMiGA. 
'''

__author__ = "Firas S Midani"
__email__ = "midani@bcm.edu"


import os
import pandas as pd
import numpy as np


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

    from string import ascii_uppercase

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


def parsename(foo):

	isolate = foo.split('_')[0]
	pm = int(foo.split('PM')[1][0])
	rep = int(foo[-1])

	meta_dict = {'Day0Anc':[0,'None'],
	             'Day20C2':[20,'None'],
	             'Day20F4':[20,'Ciprofloxacin'],
	             'Day20P1':[20,'Piperacillin'],
	             'Day20T3':[20,'Tobramycin']}

	return [foo,isolate,pm,rep]+meta_dict[isolate]



if __name__ == '__main__':

    # pointers
    supplement='./non_cdiff/dunphy/supplement'
    working='./non_cdiff/dunphy/working'

    # create amiga working folder and sub-folders, if needed
    if not os.path.exists(working):  os.makedirs(working)
    if not os.path.exists('{}/data'.format(working)): os.makedirs('{}/data'.format(working))
    if not os.path.exists('{}/mapping'.format(working)): os.makedirs('{}/mapping'.format(working))

    # read raw Biolog data
    pm1 = pd.read_csv('{}/S1A_Code/biologDataPM1.csv'.format(supplement),header=0)
    pm2 = pd.read_csv('{}/S1A_Code/biologDataPM2.csv'.format(supplement),header=0)

    pm1.columns = ['expID','PM','Strain','Time'] + list(parseWellLayout(0).index)
    pm2.columns = ['expID','PM','Strain','Time'] + list(parseWellLayout(0).index)

    # replace long plate IDs with concise amiga-compatible plate IDs

    # dictionary to count number of replicates for each strain
    strains = set(list(pm1.Strain.unique())+list(pm2.Strain.unique()))

    s_count = { s+'_PM1' : 1 for s in strains}
    s_count.update({ s+'_PM2' : 1 for s in strains})

    # dictionary to map filepath to plate_ID
    pid_dict = {}

    # for each unique filename
    for filename in list(pm1.expID.unique())+list(pm2.expID.unique()):

    	# identify strain name and pm number
    	strain = filename.split('_')[6].replace('Control','C')
    	pm = filename.split('_')[7][0:3]
    	pid = strain + '_' + pm

    	# define replicate number
    	rep = str(s_count[pid])
    	s_count[pid] += 1 

    	# complete name with hyphenated rep
    	pid = pid + '-' + rep
    	pid_dict[filename] = pid

    # replace file path in Dunphy's raw tables wiht amiga-friendly plate ID
    pm1.expID.replace(pid_dict,inplace=True)
    pm2.expID.replace(pid_dict,inplace=True)

    # drop unnecessary columns
    pm1.drop(labels=['PM','Strain'],axis=1,inplace=True)
    pm2.drop(labels=['PM','Strain'],axis=1,inplace=True)

    # concatenating PM1 and PM2 tables
    data = pd.concat([pm1,pm2])

    mapping = []
    # parse through plates and save
    for pid in data.expID.unique():

        # we don't need expID or Time and ignore index
    	data_sub = data[data.expID==pid].drop(['expID','Time'],axis=1).reset_index(drop=True).T

        # save to AMiGA working directory for Dunphy 2019 data
    	data_sub.to_csv('{}/data/{}.txt'.format(working,pid),sep='\t',header=False,index=True)

    	# pad mapping
    	mapping.append(parsename(pid))

    # create meta.txt file
    mapping = pd.DataFrame(mapping,columns=['Plate_ID','Isolate','PM','Rep','Day','Antibiotics'])
    mapping = mapping.sort_values(['Plate_ID'])
    mapping.to_csv('{}/mapping/meta.txt'.format(working),sep='\t',header=True,index=False)
