#!/usr/bin/env python

'''
AMiGA wrapper: comparing parameters between two user-defined conditions.
'''

__author__ = "Firas S Midani"
__email__ = "midani@bcm.edu"


# TABLE OF CONTENTS (6 functions)

# main
# save
# read
# validate
# subset
# compare


import sys
import pandas as pd # type: ignore

from .interface import checkParameterCommand
from .params import initParamList, articulateParameters
from .utils import subsetDf


def main(args):

	if (args.confidence < 80) | (args.confidence > 100):
		msg = '\nFATAL USER ERROR: Confdience must be between 80 and 100.\n'
		sys.exit(msg) 

	df = validate(read(args))
	df,varbs = subset(args,df)
	df = compare(args,df,varbs)
	save(args,df)


def save(args,df):

	df = articulateParameters(df,axis=0)
	df.to_csv(f'{args.output}.txt',sep='\t',header=False,index=True)


def read(args):

	ls_df = []

	for ii in args.input:
		ls_df.append(pd.read_csv(ii,sep='\t',header=0,index_col=None))

	df = pd.concat(ls_df,sort=False).reset_index(drop=True)

	return df


def validate(df):
	'''
	Do the columns include summary statistics for growth parameters? Look for 
	column headers that have the format of mean(...) and std(...).

	Args:
		df (pandas.DataFrame)

	Returns: 
		(boolean): True or False
	'''
	cols = df.keys()
	means = [True if ii.startswith('mean') else False for ii in cols]
	stds = [True if ii.startswith('std') else False for ii in cols]
	if (not any(means)) | (not any(stds)):
		msg = '\nFATAL USER ERROR: The input file does not contain summary '
		msg += 'statistics for growth parameter, such as mean and standar deviation.\n'
		sys.exit(msg)
	else:
		return df


def subset(args,df):

	ls_df, ls_varbs = [], []

	if isinstance(args.subset,str):
		args.subset = [args.subset]

	for ii in args.subset:
		criteria = checkParameterCommand(ii,sep=',')
		ls_df.append(subsetDf(df,criteria))
		ls_varbs.append(list(criteria.keys()))

	df = pd.concat(ls_df,sort=False).reset_index(drop=True).drop_duplicates()

	if df.shape[0] != 2:

		msg = '\nFATAL USER ERROR: User-provided summary files and subsetting crieteria '
		msg += 'selected for more or less than two conditions. AMiGA can only perform '
		msg += 'comparison on two conditions. Please check your arguments and try again. '
		msg += 'Below are the currently selected conditions.\n\n'

		keys = [ii for ii in df.keys() if ('(' not in ii) & (ii != 'diauxie') ]

		print(msg)
		print(df.loc[:,keys],'\n\n')
		sys.exit()

	else:

		ls_varbs = [item for sublist in ls_varbs for item in sublist]
		ls_varbs = list(set(ls_varbs))

		return df,ls_varbs


def compare(args,df,varbs):


	def getConfInts(means,stds,z_value=0.975):
		'''
		Computes confidence interval based on mean, standard deviation, and desired confidence.

		Args:
			means (array of floats)
			stds (array of floats)
			z_value (float)

		Returns:
			cis (array of strings), where each formatted string indicates the confidence interval,
			e.g. [0.4,0.6]
		'''

		from scipy.stats import norm

		scaler = norm.ppf(z_value)

		cis = []
		for m,s in zip(means,stds):
			low, upp = m-scaler*s, m+scaler*s
			cis.append(f'[{low:.3f},{upp:.3f}]')

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

	z_value = (100 - (100 - args.confidence)/2) / 100 

	if 'Sample_ID' in df.keys():
		df = df.set_index(['Sample_ID'])

	params = set(df.keys()).difference(set(varbs))
	params = list(params.intersection(initParamList(1)+['diauxie']))
	params = list({ii.split('(')[1][:-1] if '(' in ii else ii for ii in params})

	df_top = df.loc[:,varbs].reset_index(drop=True).T

	df_bot = pd.DataFrame(index=params,columns=df_top.columns).sort_index()
	df_mus = pd.concat([df_top,df_bot])

	df_bot = pd.DataFrame(index=params,columns=list(df_top.columns)+['']).sort_index()
	df_cis = pd.concat([df_top,df_bot])

	for p in params:
		if p == 'diauxie':
			df_mus.loc[p,:] = df.loc[:,p].values
			if df.loc[:,p].values[0] != df.loc[:,p].values[1]:
				df_cis.loc[p,:] = ['NA','NA',True]
			else:
				df_cis.loc[p,:] = ['NA','NA',False]
		else:
			mus = df.loc[:,f'mean({p})'].values
			stds = df.loc[:,f'std({p})'].values
			cis = getConfInts(mus,stds,z_value)
			olap = detSigDiff(eval(cis[0]),eval(cis[1]))

			df_mus.loc[p,:] = [f'{ii:.3f}' for ii in mus]
			df_cis.loc[p,:] = cis + [olap]

	df_mus.loc['Parameter',:] = ['Mean','Mean']
	df_cis.loc['Parameter',:] = [f'{args.confidence}% CI',
	                             f'{args.confidence}% CI',
	                             'Sig. Diff.']

	df_all = df_mus.join(df_cis,lsuffix='L',rsuffix='R')
	df_all = df_all.loc[varbs+['Parameter']+sorted(params),:]
	df_all = df_all.T.reset_index(drop=True).T

	return df_all

