#!/usr/bin/env python

'''
AMiGA wrapper: computing confidene interval for summary or data files.
'''

__author__ = "Firas S Midani"
__email__ = "midani@bcm.edu"


# TABLE OF CONTENTS (3 functions)

# main
# get_parameter_confidence
# get_curve_confidence


import os
import numpy as np # type: ignore
import pandas as pd # type: ignore

from .comm import smartPrint, tidyMessage
from .org import isFileOrFolder 
from .utils import flattenList

from scipy.stats import norm # type: ignore


def main(args):

	verbose = args.verbose
	directory = args.input
	overwrite = args.over_write
	confidence = float(args.confidence) / 100.0
	z_value = (1-(1-confidence)/2)
	add_noise = args.include_noise

	msg = 'AMiGA is parsing your file(s)'

	smartPrint('',verbose)
	smartPrint(tidyMessage(msg),verbose)

	directory,filename = isFileOrFolder(directory,up=1)

	# packge filename(s) into a list
	if filename:
		ls_files = [f'{directory}{os.sep}{filename}']
	else:
		ls_files = filename

	for lf in ls_files:

		df = pd.read_csv(lf,sep='\t',header=0,index_col=0)

		# define file name for the updated dataframe
		if overwrite: 
			new_name = lf
		elif lf.endswith('.txt'):
			new_name = f'{lf[:-4]}_confidence.txt'
		else:
			new_name = f'{lf}_confidence.txt'

		# compute confidecne intervals and save results
		if args.type == 'Parameters':
			df = get_parameter_confidence(df,z_value)
			df.to_csv(new_name,sep='\t',header=True,index=True)
		elif args.type == 'Curves':
			df = get_curve_confidence(df,z_value,add_noise)
			df.to_csv(new_name,sep='\t',header=True,index=False)


def get_parameter_confidence(df,z_value):
	''''Compute confidence intervals for parameters tabulated in dataframe

	Args:
		df (pandas.DataFrame): arameters must have both mean({}) and std({}) values
		confidence (float)
		add_noise (boolean): include measurement variance when computing confidence intervals

	Returns:
		df (pandas.DataFrame)
	'''

	def conf_interval(mu_std,z_value=0.975):
		mean,std = mu_std
		low = mean-norm.ppf(z_value)*std
		upp = mean+norm.ppf(z_value)*std
		return [low,upp]
	
	def getp(param):
		headers = [f"{summary}({param})" for summary in ['mean','std']]
		return row[headers].values
	
	def stats_p(param):
		headers = [f"{summary}({param})" for summary in ['mean','std','low','upp']]
		return headers

	# get all params in table with both mean and std
	keys = df.keys()
	set1 = [ii.split('mean(')[1][:-1] for ii in keys if ii.startswith('mean(')]
	set2 = [ii.split('std(')[1][:-1] for ii in keys if ii.startswith('std(')]
	params = sorted(list(set(set1).intersection(set2)))

    # define new dataframe for storing lower and upper confidnce intervals
	ci_keys =flattenList([[f'low({ii})',f'upp({ii})'] for ii in params])
	ci_df = pd.DataFrame(index=df.index,columns=ci_keys)

    # for each sample/row, compute and store confidence intervals
	for idx,row in df.iterrows():
		ci_df.loc[idx,ci_keys] = flattenList([conf_interval(getp(ii),z_value) for ii in params])

	# order columns by grouping stats of each parameter
	order_params = (flattenList([stats_p(pp) for pp in params]))
	non_params = list(set(keys).difference(set(order_params)))

	# put together the dataframe
	df = df.join(ci_df).loc[:,non_params+order_params]

	return df

def get_curve_confidence(df,z_value,add_noise=False):
	'''Compute confidence intervals for growth and growth rate functions

	Args:
		df (pandas.DataFrame): columns must include mu, Sigma, mu1, Sigma1, Noise
		confidence (float)
		add_noise (boolean): include measurement variance when computing confidence intervals

	Returns:
		df (pandas.DataFrame)
	'''

	def conf_interval(mu,sigma,noise,z_value,add_noise=add_noise):
		if add_noise:
			sigma = sigma + noise
		else:
			sigma = sigma
		band = norm.ppf(z_value)*np.sqrt(sigma)
		low = list((mu-band).values)
		upp = list((mu+band).values)
		return [low,upp]

    # what are original keys & create df for storing intervals
	keys = df.keys()
	ci_df = pd.DataFrame(index=df.index,columns=['low','upp','low1','upp1'])

    # get confidence intervals for grwoth and growth rate
	foo0 = conf_interval(df['mu'],df['Sigma'],df['Noise'],z_value=z_value,add_noise=add_noise)
	foo1 = conf_interval(df['mu1'],df['Sigma1'],None,z_value=z_value,add_noise=False)

    # store in dataframe
	ci_df.loc[:,'low'] = foo0[0]
	ci_df.loc[:,'upp'] = foo0[1]
	ci_df.loc[:,'low1'] = foo1[0]
	ci_df.loc[:,'upp1'] = foo1[1]

    # reorder parameters
	order_params = ['mu','low','upp','Sigma','mu1','low1','upp1','Sigma1','Noise']
	non_params = list(set(keys).difference(set(order_params)))

	# put together the dataframe
	df = df.join(ci_df).loc[:,non_params+order_params]

	return df

