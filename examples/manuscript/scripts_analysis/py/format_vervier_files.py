#!/usr/bin/env python

'''
Script for parsing sample data from CarboLogR and formatting them to be
compatible with AMiGA. 
'''

__author__ = "Firas S Midani"
__email__ = "midani@bcm.edu"


import os
import pandas as pd
import numpy as np


def tabulateLine(line):
    '''Turns data read as a line into growth ucrve tables (wells x time)'''

    ls = []
    for ii in line.split('Hour')[1].strip('\t').strip().split('\n'):
        items = [jj.strip() for jj in ii.split('\t')]
        if items[0] == 'A01':
            items = ['Hour'] + items
        ls.append(items)

    df = pd.DataFrame(ls[1:],columns=ls[0])
    df = df.set_index('Hour').T

    df.columns = [float(ii) for ii in df.columns]

    return df


def adjustWellIDs(df):
	'''Example A01 to A1'''

	df.index = [ii[0] + str(int(ii[1:])) for ii in df.index]

	return df


if __name__ == '__main__':

    # pointers
    carbologr = './non_cdiff/vervier/CarboLogR/inst/extdata/exampleDataPM1'
    working='./non_cdiff/vervier/working'

    # create amiga working folder and sub-folders, if needed
    if not os.path.exists(working):  os.makedirs(working)
    if not os.path.exists('{}/data'.format(working)): os.makedirs('{}/data'.format(working))
    if not os.path.exists('{}/mapping'.format(working)): os.makedirs('{}/mapping'.format(working))

    ls_files = [ii for ii in os.listdir(carbologr) if ii.endswith('.csv')]

    mapping = []
    for lf in ls_files:

        fid = open('{}/{}'.format(carbologr,lf),'r')
        line = fid.read()
        line = line.replace(',','\t')
        df = tabulateLine(line)
        df = adjustWellIDs(df)
        fid.close()

        isolate = lf.split('_')[0]
        rep = lf.split('_')[-1][0]

        filebase = '{}_PM1-{}'.format(isolate,rep)

        df.to_csv('./non_cdiff/vervier/working/data/{}.txt'.format(filebase),sep='\t',header=True,index=True)

        mapping.append([filebase,isolate,int(1),int(rep)])

    # create meta.txt file
    mapping = pd.DataFrame(mapping,columns=['Plate_ID','Isolate','PM','Rep'])
    mapping.to_csv('{}/mapping/meta.txt'.format(working),sep='\t',header=True,index=False)
