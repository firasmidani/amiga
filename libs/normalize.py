#!/usr/bin/env python

'''
AMiGA wrapper: normalizing parameters in summary files.
'''

__author__ = "Firas S Midani"
__email__ = "midani@bcm.edu"


# TABLE OF CONTENTS (# functions)


import os
import sys
import argparse
import operator
import pandas as pd

from libs.comm import smartPrint, tidyMessage
from libs.interface import checkParameterCommand
from libs.org import isFileOrFolder 
from libs.read import findPlateReaderFiles
from libs.params import initParamList
from libs.utils import subsetDf


def main(args):

	verbose = args.verbose
	directory = args.input
	ovewrrite = args.over_write

	msg = 'AMiGA is parsing your file(s)'

	smartPrint('',verbose)
	smartPrint(tidyMessage(msg),verbose)
	
	directory,filename = isFileOrFolder(directory,up=1)

	if filename: ls_files = ['{}{}{}'.format(directory,os.sep,filename)]
	else: ls_files = findPlateReaderFiles(directory,'.txt')

	for lf in ls_files:

		df = read(ls_files)

		if ovewrrite:  new_name = lf
		elif lf.endswith('.txt'): new_name = '{}_normalized.txt'.format(lf[:-4])
		else: new_name = '{}.normalized.txt'.format(lf)
		
		df = normalizeParameters(args,df)
		df.to_csv(new_name,sep='\t',header=True,index=True)


def read(ls_fpaths):
	'''
	Reads all files listed in input argument and concats tables
	    into one large pandas.DataFrame.

	Args:
	    ls_fpaths (list)

	Return:
	    pandas.DataFrame
	'''

	df_list = []

	for fpath in ls_fpaths:
		df_list.append(pd.read_csv(fpath,sep='\t',header=0,index_col=0))

	df = pd.concat(df_list,sort=False)

	return df


def normalizeParameters(args,df):
	'''
	Normalizes growth parameters to control samples. 

	Args:
	    args (dictionary): keys are arguments and value are user/default choices
	    df (pandas.DataFrame): rows are samples, columns are experimental variables. Must include
	        Plate_ID, Group, Control, auc, k, gr, dr, td, lag.

	Returns:
	    df (pandas.DataFrame): input but with an additional 6 columns.
	'''

	# let's keep original dataframe
	df_orig = df.copy()
	df_orig_keys = df_orig.columns 
	df = df.reset_index()

	# How you should normalize?
	if args.normalize_method == 'division':  opr = operator.truediv
	elif args.normalize_method == 'subtraction':  opr = operator.sub

	# How to group samples and which ones are control samples?

	# if user specifies with command-line arguments
	if args.group_by is not None or args.control_by is not None:

		groupby = args.group_by.split(',')
		controlby = checkParameterCommand(args.control_by)

	# else check columns for Group and Contol variables
	elif 'Group' in df_orig_keys and 'Control' in df_orig_keys:

		groupby = ['Group']
		controlby = {'Control':1}

    # else exit with error message
	else:
		msg = 'FATAL USER ERROR: User must specify groups of samples and '
		msg += 'their corresponding control samples.'
		sys.exit(msg)

	# which parameters to normalize and/or to keep
	params_1 = initParamList(0)
	params_1.remove('diauxie')
	params_2 = ['mean({})'.format(ii) for ii in params_1]

	if any([ii in df_orig_keys for ii in params_2]):  params = params_2
	else:  params = params_1

	params_norm = initParamList(2)
	params_keep = groupby + list(controlby.keys()) + ['Sample_ID','PlateID'] + params
	params_keep = list(df.columns[df.columns.isin(params_keep)])
	params_varbs = list(set(params_keep).difference(set(params)))
	df = df.loc[:,params_keep]

	norm_df = []
	for idx,row in df.loc[:,groupby].drop_duplicates().iterrows():

		df_group = subsetDf(df,row.to_dict()).loc[:,params_keep]
		df_group = df_group.sort_values(params_varbs)
		df_control = subsetDf(df_group,controlby)


		df_group.set_index(params_varbs,inplace=True)
		df_control.set_index(params_varbs,inplace=True)

		dgv = df_group.values
		dcv = df_control.values

		df_group.loc[:,:] = opr(dgv,dcv)
		norm_df.append(df_group)	

	norm_df = pd.concat(norm_df,axis=0)
	norm_df.columns = params_norm	
	norm_df = norm_df.reset_index(drop=False)

	df = pd.merge(df_orig,norm_df,on=params_varbs)

	if 'Sample_ID' in df.columns: df = df.set_index('Sample_ID')

	return df

