#!/usr/bin/env python

'''
AMiGA wrapper: getting time at which measurements reach a user-defined threshold.
'''

__author__ = "Firas S Midani"
__email__ = "midani@bcm.edu"


# TABLE OF CONTENTS (4 functions)

# main
# update
# find
# read

import numpy as np # type: ignore
import pandas as pd # type: ignore


def main(args):

	tabs = read(args)
	summ = find(args,tabs)
	summ = update(args,summ)


def update(args,summ):

	summ.to_csv(args.summary,sep='\t',header=True,index=False)

def find(args,tables):

    # define arguments
	col = args.curve_format
	thresh = args.threshold

	data, summ = tables

	ls_time = []
	sample_ids = summ.Sample_ID.unique()

	for sid in sample_ids:

		tmp = data[data.Sample_ID==sid]
		tmp = tmp.loc[:,['Time',col]]

		matches = tmp[tmp[col]>=thresh].Time.values

		if len(matches) == 0:
			ls_time.append(np.inf)
		else:
			ls_time.append(matches[0])

	description = f't(od>={thresh})'
	t_od = pd.DataFrame([sample_ids,ls_time],index=['Sample_ID',description]).T

	summ = pd.merge(summ,t_od,on='Sample_ID')

	return summ


def read(args):

	data = pd.read_csv(args.gp_data,sep='\t',index_col=0)
	summ = pd.read_csv(args.summary,sep='\t')

	tables = (data,summ)

	return tables

