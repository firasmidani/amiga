#!/usr/bin/env python

'''
AMiGA wrapper: normalizing parameters in summary files.
'''

__author__ = "Firas S Midani"
__email__ = "midani@bcm.edu"


# TABLE OF CONTENTS (3 functions)

# main
# read
# normalize_parameters


import os
import sys
import operator
import pandas as pd # type: ignore

from .comm import smartPrint, tidyMessage
from .interface import checkParameterCommand
from .org import isFileOrFolder 
from .read import findPlateReaderFiles
from .params import initParamList
from .utils import subsetDf


def main(args):

	verbose = args.verbose
	directory = args.input
	ovewrrite = args.over_write

	msg = 'AMiGA is parsing your file(s)'

	smartPrint('',verbose)
	smartPrint(tidyMessage(msg),verbose)
	
	directory,filename = isFileOrFolder(directory,up=1)

	if filename:
		ls_files = [f'{directory}{os.sep}{filename}']
	else:
		ls_files = findPlateReaderFiles(directory,'.txt')

	for lf in ls_files:

		df = read(ls_files)

		if ovewrrite: 
			new_name = lf
		elif lf.endswith('.txt'):
			new_name = f'{lf[:-4]}_normalized.txt'
		else:
			new_name = f'{lf}.normalized.txt'
		
		df = normalizeParameters(args,df)
		df.to_csv(new_name,sep='\t',header=True,index=True)

	msg = 'AMiGA compelted your request'

	smartPrint('',verbose)
	smartPrint(tidyMessage(msg),verbose)


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
	if args.normalize_method == 'division':
		opr = operator.truediv
	elif args.normalize_method == 'subtraction':
		opr = operator.sub

	# How to group samples and which ones are control samples?

	# if user specifies with command-line arguments
	if args.group_by is not None and args.normalize_by is not None:

		groupby = args.group_by.split(',')
		controlby = checkParameterCommand(args.normalize_by)

	elif args.normalize_by is not None and args.group_by is None:

		controlby = checkParameterCommand(args.normalize_by)
		df.loc[:,'Group'] = [1]*df.shape[0]
		groupby = ['Group']

	# else check columns for Group and Contol variables
	elif 'Group' in df_orig_keys and 'Control' in df_orig_keys:

		groupby = ['Group']
		controlby = {'Control':1}

		if (len(df.Group.unique())==1) and (len(df.Plate_ID.unique())>1):
			msg = '\nUSER WARNING: AMiGA detected a single "Group" but multiple Plate_IDs.\n'
			msg += 'Wells from different plates will thus be normalized togther as a group.\n'
			msg += 'If this was not your intention, please pass explicit arguments to AMiGA\n'
			msg += 'using "--group-by" and "--control-by" arguments to avoid any ambiguity.\n' 
			print(msg)

    # else exit with error message
	else:
		msg = 'FATAL USER ERROR: User must specify groups of samples and '
		msg += 'their corresponding control samples.'
		sys.exit(msg)

	# which parameters to normalize and/or to keep
	params_1 = initParamList(0)
	params_1.remove('diauxie')
	params_2 = [f'mean({ii})' for ii in params_1]
	params_3 = initParamList(2)

	if any([ii in df_orig_keys for ii in params_2]):  
		params = params_2
	elif any([ii in df_orig_keys for ii in params_3]):
		params = params_3 
	else:
		params = params_1

	#params_norm = initParamList(2)

	params_keep = groupby + list(controlby.keys()) + ['Sample_ID','Plate_ID'] + params
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
		dcv = df_control.mean(numeric_only=True).values

		df_group.loc[:,:] = opr(dgv,dcv)
		norm_df.append(df_group)	

	norm_df = pd.concat(norm_df,axis=0)
	norm_df.columns = [f'norm({ii})' for ii in norm_df.columns]	
	norm_df = norm_df.reset_index(drop=False)

	df = pd.merge(df_orig,norm_df,on=params_varbs)

	if 'Sample_ID' in df.columns:
		df = df.set_index('Sample_ID')

	return df

