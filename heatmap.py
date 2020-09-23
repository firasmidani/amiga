#!/usr/bin/env python

'''
AMiGA wrapper: plotting heatmaps.
'''

__author__ = "Firas S Midani"
__email__ = "midani@bcm.edu"


# TABLE OF CONTENTS (8 functions)

# main
# read
# reduceDf
# group
# pivot
# clusterMap
# plot
# parseCommand

import os
import argparse
import operator
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from functools import reduce
from matplotlib import rcParams

rcParams.update({'font.size':20})
sns.set_style('whitegrid')

from libs.comm import smartPrint, tidyDictPrint, tidyMessage
from libs.interface import checkParameterCommand,integerizeDictValues
from libs.org import assemblePath, assembleFullName
from libs.org import isFileOrFolder, checkDirectoryNotEmpty, printDirectoryContents
from libs.read import findPlateReaderFiles
from libs.utils import selectFileName, subsetDf


def main():

	args = parseCommand()
	verbose = args['verbose']
	#directory = assemblePath(args['fi'],'summary')
	directory = args['fi']

	msg = 'AMiGA is peeking inside the summary directory'

	smartPrint('',verbose)
	smartPrint(tidyMessage(msg),verbose)
	# smartPrint(checkDirectoryNotEmpty(directory,'Summary')[1],verbose)

	criteria = checkParameterCommand(args['s'],sep=',')
	
	directory,filename = isFileOrFolder(directory,up=1)

	if filename: ls_files = ['{}{}{}'.format(directory,os.sep,filename)]
	else: ls_files = findPlateReaderFiles(directory,'.txt')

	df = read(ls_files)
	df = subsetDf(df,criteria)
	df = group(df,args)
	df = pivot(df,args)
	df = reduceDf(df,args)
	#plot(df,args,directory)
	clusterMap(df,args,directory)


def read(ls_fpaths):

	df_list = []

	for fpath in ls_fpaths:
		df_list.append(pd.read_csv(fpath,sep='\t',header=0,index_col=0))

	df = pd.concat(df_list,sort=False)

	return df


def reduceDf(df,args):

	if args['f'] is None:
		return df

	if 'OR' in args['f']:
		cmds = args['f'].split('OR')
	else:
		cmds = [args['f']]

	ls_df = []

	for cc in cmds:
		axis,outer,op,thresh = cc.strip().split(' ')

		# define logical operator
		delimiters = {'>=':operator.ge,
		              '<=':operator.le,
		              '>':operator.gt,
		              '<':operator.lt,
		              '=':operator.eq}
		op = delimiters[op]  

		# define outer operation
		outer_dict = {'any':any,'all':all,'':all}
		outer = outer_dict[outer]

		# define axis for outer operation
		axis_dict={'row':1,'col':0}
		axis = axis_dict[axis]
		
		ls_df.append(df[df.apply(lambda x: outer(op(x,float(thresh))),axis=axis)])

	df = pd.concat(ls_df,axis=0,sort=False)

	return df


def group(df,args):

	opr = args['p']

	if opr is None: #opr = 'mean'
		return df

	df = df.groupby([args['x'],args['y']])

	if opr == 'mean':
		df = df.mean()
	elif opr == 'median':
		df = df.median()

	df = df.reset_index()

	return df


def pivot(df,args):

	df = pd.pivot(data=df,columns=args['x'],index=args['y'],values=args['v'])

	return df


def clusterMap(df,args,directory):

	def dekwarg(ii):
		key = ii.split(':')[0]
		value = ii.split(':')[1]
		if value.replace('.','',1).isdigit():
			value = float(value)
		elif value in ['True','False']:
			value = bool(value)
		return key,value

	ny,nx = df.shape
	figsize=[nx*2+6,ny*0.5+3]

	kwargs = {'row_cluster':False,'col_cluster':False,'figsize':figsize}

	if args['kwargs']:
		h_kwargs = args['kwargs'].split(';')
		h_kwargs = [dekwarg(ii) for ii in h_kwargs]
		h_kwargs = {k:v for k,v in h_kwargs}
		kwargs.update(h_kwargs)
	
	c = sns.clustermap(df,**kwargs)

	kwargs = {'xlabel':'',
		      'ylabel':'',
		      'title':[args['v'] if args['t'] is None else args['t']][0]}

	c.ax_heatmap.set(**kwargs)
	c.ax_row_dendrogram.set_visible(False)

	dendro_box = c.ax_row_dendrogram.get_position()
	dendro_box.x0 = (dendro_box.x0 + 2 * dendro_box.x1) /3
	dendro_box.x0 = dendro_box.x0 - 0.01
	dendro_box.x1 = dendro_box.x1 - 0.01
	c.cax.set_position(dendro_box)
	c.cax.yaxis.set_ticks_position("left")

	[ii.set(fontsize=30) for ii in c.ax_heatmap.get_xticklabels()+c.ax_heatmap.get_yticklabels()]
	[ii.set(rotation=90) for ii in c.ax_heatmap.get_xticklabels()]

	#cbar = ax.collections[0].colorbar
	#cbar.ax.tick_params(labelsize=20)

	#fpath = assembleFullName(args['fi'],'',args['fo'],'clustered','.pdf')
	fpath = assembleFullName(directory,'',args['fo'],'','.pdf')
	plt.savefig(fpath,bbox_inches='tight')


def plot(df,args,directory):

	def dekwarg(ii):
		key = ii.split(':')[0]
		value = ii.split(':')[1]
		if value.replace('.','',1).isdigit():
			value = float(value)
		elif value in ['True','False']:
			value = bool(value)
		return key,value

	ny,nx = df.shape
	figsize=[nx*2,ny*0.5]

	kwargs = {}

	if args['kwargs']:
		h_kwargs = args['kwargs'].split(';')
		h_kwargs = [dekwarg(ii) for ii in h_kwargs]
		h_kwargs = {k:v for k,v in h_kwargs}

		kwargs.update(h_kwargs)

	#fig,ax = plt.subplots()#figsize=[nx*4,ny*0.5])
	fig,ax = plt.subplots(figsize=figsize)

	sns.heatmap(df,ax=ax,**kwargs)

	kwargs = {'xlabel':'',
		      'ylabel':'',
		      'title':[args['v'] if args['t'] is None else args['t']][0]}

	ax.set(**kwargs)

	[ii.set(rotation=90) for ii in ax.get_xticklabels()]

	cbar = ax.collections[0].colorbar
	cbar.ax.tick_params(labelsize=20)

	#dpath = assemblePath(args['fi'],'summary')
	#fpath = assembleFullName(args['fi'],'',args['fo'],'','.pdf')
	fpath = assembleFullName(directory,'',args['fo'],'','.pdf')
	plt.savefig(fpath,bbox_inches='tight')


def parseCommand():
    '''
    Interprets the arguments passed by the user to AMiGA. 

    Note: Function is AMiGA-specific and should not be used verbatim for other apps.

    Returns:
        args (dict): a dictionary with keys as suggested variable names
            and keys as the user-passed and argparse-interpreted arguments.
    '''

    args_dict= {};

    # parse arguments 
    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--input',required=True)
    parser.add_argument('-o','--output',required=True)
    parser.add_argument('-s','--subset',required=False)
    parser.add_argument('-v','--value',required=True)
    parser.add_argument('-x','--x-variable',required=True)
    parser.add_argument('-y','--y-variable',required=True)
    parser.add_argument('-p','--operation',required=False)
    parser.add_argument('-f','--filter',required=False)
    parser.add_argument('-t','--title',required=False)
    parser.add_argument('--kwargs',required=False)
    parser.add_argument('--verbose',action='store_true',default=False)

    # pass arguments to local variables 
    args = parser.parse_args()
    args_dict['fi'] = args.input  # File path provided by user
    args_dict['fo'] = args.output
    args_dict['s'] = args.subset
    args_dict['x'] = args.x_variable
    args_dict['y'] = args.y_variable
    args_dict['v'] = args.value
    args_dict['p'] = args.operation
    args_dict['f'] = args.filter
    args_dict['t'] = args.title
    args_dict['kwargs'] = args.kwargs
    args_dict['verbose'] = args.verbose

    # summarize command-line artguments and print
    if args_dict['verbose']:
        msg = '\n'
        msg += tidyMessage('User provided the following command-line arguments:')
        msg += '\n' 
        msg += tidyDictPrint(args_dict)
        print(msg)

    return args_dict


if __name__ == "__main__":

	main()