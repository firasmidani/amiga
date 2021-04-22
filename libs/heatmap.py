#!/usr/bin/env python

'''
AMiGA wrapper: plotting heatmaps.
'''

__author__ = "Firas S Midani"
__email__ = "midani@bcm.edu"


# TABLE OF CONTENTS (9 functions)

# read
# clusterMap
# generate_missing_color
# getColorLegend
# group
# main
# pivot
# reduceDf
# saveDf

import os
import random
import argparse
import operator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from functools import reduce
from matplotlib import rcParams
from matplotlib.colors import to_rgb

rcParams.update({'font.size':20})
sns.set_style('whitegrid')

from libs.comm import smartPrint, tidyDictPrint, tidyMessage
from libs.interface import checkParameterCommand,integerizeDictValues
from libs.org import assemblePath, assembleFullName
from libs.org import isFileOrFolder, checkDirectoryNotEmpty, printDirectoryContents
from libs.params import initParamList
from libs.read import findPlateReaderFiles
from libs.utils import flattenList, selectFileName, subsetDf


def main(args):

	verbose = args.verbose
	#directory = assemblePath(args.input,'summary')
	directory = args.input

	msg = 'AMiGA is peeking inside the summary directory'

	smartPrint('',verbose)
	smartPrint(tidyMessage(msg),verbose)
	# smartPrint(checkDirectoryNotEmpty(directory,'Summary')[1],verbose)

	criteria = checkParameterCommand(args.subset,sep=',')
	
	directory,filename = isFileOrFolder(directory,up=1)

	if filename: ls_files = ['{}{}{}'.format(directory,os.sep,filename)]
	else: ls_files = findPlateReaderFiles(directory,'.txt')

	full_df = read(ls_files)
	sub_df = subsetDf(full_df,criteria)
	sub_df = group(sub_df,args)
	sub_df = pivot(sub_df,args,args.value)
	sub_df = reduceDf(sub_df,args)
	clusterMap(sub_df,full_df,args,directory)
	saveDf(full_df,sub_df,args,directory)


def saveDf(full_df,sub_df,args,directory):

	if not args.save_filtered_table:
		return None

	sub_df = subsetDf(full_df,{args.y_variable:list(sub_df.index.values),
		                       args.x_variable:list(sub_df.keys().values)})

	fpath = assembleFullName(directory,'',args.output,'filtered','.txt')
	sub_df.to_csv(fpath,sep='\t',header=True,index=True)

def read(ls_fpaths):

	df_list = []

	for fpath in ls_fpaths:
		foo = pd.read_csv(fpath,sep='\t',header=0,index_col=0,low_memory=False)
		df_list.append(foo)
	df = pd.concat(df_list,sort=False)

	return df


def reduceDf(df,args):

	if args.filter is None:
		return df

	if 'OR' in args.filter:
		cmds = args.filter.split('OR')
	else:
		cmds = [args.filter]

	ls_rows = []
	ls_cols = []

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
		axis_dict={'row':int(1),'col':int(0)}
		axis = axis_dict[axis]

		if axis==1:
			foo = df.loc[df.apply(lambda x: outer(op(x,float(thresh))),axis=axis),:]
			ls_rows.append(list(foo.index.values))
			ls_cols.append(list(foo.columns))
		elif axis==0:
			foo = df.loc[:,df.apply(lambda x: outer(op(x,float(thresh))),axis=axis)]
			ls_rows.append(list(foo.index.values))
			ls_cols.append(list(foo.columns))

		#ls_df.append(foo)
		#ls_df.append(df[df.apply(lambda x: outer(op(x,float(thresh))),axis=axis)])

	ls_rows = sorted(list(set(flattenList(ls_rows))))
	ls_cols = sorted(list(set(flattenList(ls_cols))))

	df = df.loc[ls_rows,ls_cols]

	#df = pd.concat(ls_df,axis=0,sort=False)

	return df


def group(df,args):

	opr = args.operation

	if opr is None: #opr = 'mean'
		return df

	df = df.groupby([args.x_variable,args.y_variable])

	if opr == 'mean':
		df = df.mean()
	elif opr == 'median':
		df = df.median()

	df = df.reset_index()

	return df

def generate_missing_color(colors):
	'''Credit to Andreas Dewes

	https://gist.github.com/adewes/5884820
	'''	

	def get_random_color(pastel_factor = 0.5):
	    return [(x+pastel_factor)/(1.0+pastel_factor) for x in [random.uniform(0,1.0) for i in [1,2,3]]]

	def color_distance(c1,c2):
	    return sum([abs(x[0]-x[1]) for x in zip(c1,c2)])

	def generate_new_color(existing_colors,pastel_factor = 0.5):
	    max_distance = None
	    best_color = None
	    for i in range(0,100):
	        color = get_random_color(pastel_factor = pastel_factor)
	        if not existing_colors:
	            return color
	        best_distance = min([color_distance(color,c) for c in existing_colors])
	        if not max_distance or best_distance > max_distance:
	            max_distance = best_distance
	            best_color = color
	    return best_color

	colors = [to_rgb(ii) for ii in colors]
	best_color = tuple(generate_new_color(colors))

	return best_color


def pivot(df,args,metric=None):

	if metric is None:

		return df

	else: 

		df = pd.pivot(data=df,columns=args.x_variable,index=args.y_variable,values=metric)

		rows_todrop = np.where(df.isna().any(1))[0]
		rows_todrop = df.index.values[rows_todrop]

		cols_todrop = np.where(df.isna().any())[0]
		cols_todrop = df.keys().values[cols_todrop]

		if len(rows_todrop) > 0 or len(cols_todrop): 
			msg = 'User Warning: The heatmap data is missing values. '
			msg += 'Pleae check the data for the following:\n\n'
			msg += 'Columns:\t'
			msg += ', '.join(cols_todrop) + '\n'
			msg += '\n'
			msg += 'Rows:\t'
			msg += ', '.join(rows_todrop) + '\n'
			msg += '\nThese variables will be dropped and not plotted unless if you requested that '
			msg += 'they be kept with --keep-rows-missing-data or --keep-columns-missing-data.\n\n'
			smartPrint(msg,args.verbose)

		if not args.keep_rows_missing_data:
			df = df.drop(labels=rows_todrop,axis=0)

		if not args.keep_columns_missing_data:
			df = df.drop(labels=cols_todrop,axis=1)

	return df



def get_color_legend(df,full_df,args,directory,axis='y'):

    # e.g. color by ribotype
    colorby = eval('args.color_'+axis+'_by')
    if colorby is None: return None

    # variable on axis
    variable = eval('args.'+axis+'_variable')

    # e.g. file with two columns: ribotype and color
    colorfile =eval('args.color_file_'+axis)

    # dictionary arguments passed by user
    colorscheme = checkParameterCommand(eval('args.color_scheme_'+axis))

    # can't pass both file and command-line argument
    if colorfile is not None and colorscheme is not None:
    	msg = 'WARNING: User must pass eithe color file or color scheme '
    	msg += ' for the {}-axis, not both.' .format(axis)
    	sys.exit(msg.format(axis))

    if colorfile is not None:
    	colors_df = pd.read_csv(colorfile,sep='\t',header=0,index_col=0)
    if colorscheme is not None:
    	colors_df = pd.DataFrame(colorscheme,index=['Color']).T

    # create list of colors based on meta-data
    if args.missing_color is None:
    	missing_color = generate_missing_color(list(colors_df.Color.values))
    else:
    	missing_color = args.missing_color

    if colorby == variable:
    	foo = full_df.loc[:,[variable]]
    else:
    	foo = full_df.loc[:,[colorby,variable]]
    
    foo = foo.drop_duplicates().set_index(variable).astype(str)
    foo = foo.join(colors_df,on=colorby)
    foo.Color = [missing_color if str(ii)=='nan' else ii for ii in list(foo.Color.values)]
    if axis == 'x': colors = foo.loc[df.columns,'Color'].values
    if axis == 'y': colors = foo.loc[df.index.values,'Color'].values

    # create legend patches
    colors_df.loc['~other~','Color'] = missing_color
    colors_df = colors_df[colors_df.Color.isin(colors)]
    colors_df = colors_df.to_dict()['Color']
    patches = [mpatches.Patch(color=color,label=label) for label,color in colors_df.items()]

    # plot legend
    fig,ax = plt.subplots(figsize=[4,4])
    ax.axis(False)
    lgd = ax.legend(handles=patches,loc='center')

    # save legend
    fpath = assembleFullName(directory,'',args.output,axis+'_legend','.pdf')
    plt.savefig(fpath,bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.close()

    return colors


def clusterMap(df,full_df,args,directory):

	def dekwarg(ii):
		'''Defines how to dfine types of user-defined arguments'''
		key = ii.split(':')[0]
		value = ii.split(':')[1]
		for r in (('.',''),('-','')):
			adj_value = value.replace(*r)
		if adj_value.isdigit():
			value = float(value)
		elif value in ['True','False']:
			value = bool(value)
		return key,value

	# define figure size (inches)
	ny,nx = df.shape

	if args.width_height is None:
		figsize=[nx*2+6,ny*0.5+3]
	else:
		w,h = args.width_height
		figsize=[float(w),float(h)]

    # package argumtns into a dictionary
	kwargs = {'row_cluster':False,'col_cluster':False,'figsize':figsize}
	if args.kwargs:
		h_kwargs = args.kwargs.split(';')
		h_kwargs = [dekwarg(ii) for ii in h_kwargs]
		h_kwargs = {k:v for k,v in h_kwargs}
		kwargs.update(h_kwargs)

	# get colors for side bars
	row_colors = get_color_legend(df,full_df,args,directory,axis='y')
	col_colors = get_color_legend(df,full_df,args,directory,axis='x')

    # clustermap
	c = sns.clustermap(df,**kwargs,dendrogram_ratio=args.colorbar_ratio,
		row_colors=row_colors,col_colors=col_colors)

	# adjust title
	title = [args.value if args.title is None else args.title][0]
	if args.color_x_by is not None:
		pad = 40
	else:
		pad = 15
	c.ax_heatmap.set_title(title,fontsize=args.fontsize,pad=pad)

	# adjust labels
	kwargs = {'xlabel':'','ylabel':''}
	c.ax_heatmap.set(**kwargs)
	c.ax_row_dendrogram.set_visible(False)
	c.ax_col_dendrogram.set_visible(False)

	# adjust color bar position and dimensions
	dendro_box = c.ax_row_dendrogram.get_position()
	dendro_box.x0 = (dendro_box.x0 + 2 * dendro_box.x1) /3
	dendro_box.x0 = dendro_box.x0 - 0.01
	dendro_box.x1 = dendro_box.x1 - 0.01
	c.cax.set_position(dendro_box)
	c.cax.yaxis.set_ticks_position("left")

	# adjust tick labels
	if (int(args.x_rotation) % 90)==0:  ha = 'center'
	else: ha = 'right'

	[ii.set(fontsize=args.fontsize) for ii in c.ax_heatmap.get_xticklabels()+c.ax_heatmap.get_yticklabels()]
	[ii.set(rotation=args.x_rotation,ha=ha) for ii in c.ax_heatmap.get_xticklabels()]
	c.cax.tick_params(labelsize=args.fontsize)

	# check for proper rendering of tick labels
	msg = 'WARNING: figure size is too small and /or fontsize is too large '
	msg += 'enought to display all {}-axis labels. Pleas increase the {} '
	msg += 'argument and/or decrease the fontsize to insure that all labels '
	msg += 'are propely printed'

	yticklabels = c.ax_heatmap.get_yticklabels()
	xticklabels = c.ax_heatmap.get_xticklabels()

	if df.shape[0] != len(yticklabels): print(msg.format('y','height'))
	if df.shape[1] != len(xticklabels): print(msg.format('x','width'))

	# highlight feature
	highlight_labels = checkParameterCommand(args.highlight_labels)

	if highlight_labels is not None:
		for key,values in highlight_labels.items():
			if key == 'x': tlabels = xticklabels
			elif key == 'y': tlabels = yticklabels
			for label in values:
				matches = np.where([ii.get_text()==label for ii in tlabels])[0]
				for match in matches: tlabels[match].set(color='red',fontweight='bold')
  

	fpath = assembleFullName(directory,'',args.output,'','.pdf')
	plt.savefig(fpath,bbox_inches='tight')