#!/usr/bin/env python

# __author__ = Firas S Midani
# __email__ = midani@bcm.edu

# Midani et al. (2020) Supp. Figure 5 is generated by this script 

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from matplotlib.ticker import MaxNLocator
from scipy.stats import norm

sns.set_style('whitegrid')

confidence = 0.975 #0.995

working = './non_cdiff/dunphy/working'

# read mapping file and identify unique strains
mapping = pd.read_csv('{}/mapping/dunphy_glcnac_mutants.txt'.format(working),sep='\t',header=0,index_col=0)
strains = list(set(mapping[mapping.Strain!='Ancestor'].Strain.values))

# order in Fig. 4B of Dunphy et al. (2019 Cell Metabolism)
dunphy_order_strains = ['Ancestor','PA14_23470','PA14_57570','nuoB','PA14_41710',
                'PA14_44360','PA14_57880','nuoL','PA14_57850']
dunphy_order = ['Ancestor_vs_{}'.format(ii) for ii in dunphy_order_strains]

###############################
# Prep Functional Differences #
###############################

# read amiga hypothesis testing results
models = []
for strain in strains:
	foo = '{}/models/Ancestor_vs_{}/Ancestor_vs_{}_log.txt'.format(working,strain,strain)
	foo = pd.read_csv(foo,sep='\t',header=0,index_col=0)
	models.append(foo)
models = pd.concat(models)

# create dummy mode comparing ancestor to itself, helpful for plotting below
models.loc['Ancestor_vs_Ancestor',['Func_Diff_Mean','Func_Diff_CI']] = [0., '(0.,0.)']

# create minimal dataframe for summary of functional differences
funcdiffs = models.loc[:,['Func_Diff_Mean','Func_Diff_CI']]
funcdiffs.loc[:,'Line'] = [ii.split('_vs_')[1] for ii in funcdiffs.index.values]
funcdiffs = funcdiffs.loc[dunphy_order,:]

############################
# Prep GP Predicted Curves #
############################

gp_data = '{}/derived/dunphy_glcnac_mutants_pooled_gp_data.txt'.format(working)
gp_data = pd.read_csv(gp_data,sep='\t',header=0,index_col=0)

def getMeanBands(foo):

	scaler = norm.ppf(confidence)

	x_time = foo.Time.values
	y_avg = foo.mu.values
	y_low = y_avg - scaler * np.sqrt(foo.Sigma.values + foo.Noise.values)
	y_upp = y_avg + scaler * np.sqrt(foo.Sigma.values + foo.Noise.values)


	return {'x':x_time, 'y':y_avg, 'y1':y_low, 'y2':y_upp}

###################
# Prep Parameters #
###################

# read parameter summary
params = '{}/summary/dunphy_glcnac_mutants_pooled_summary.txt'.format(working)
params = pd.read_csv(params,sep='\t',header=0,index_col=0)
params.set_index('Strain',inplace=True)

# reduce to three parameters
keep_keys = ['mean({})'.format(ii) for ii in ['k_log','gr','lagC']]
keep_keys += ['std({})'.format(ii) for ii in ['k_log','gr','lagC']]
params = params.loc[:,keep_keys]

def detSigDiff(a,b):
	'''Check if confidence intervals overlap or not. Return True if no overlap.'''
	a = [float(ii) for ii in a]
	b = [float(ii) for ii in b]
	return not ((a[0] <= b[1]) and (b[0] <= a[1]))

def conf_int(conf=.975,std=None,m=None):
    '''Compute 95% confidence interval.'''
    
    from scipy.stats import norm
    
    h = std*norm.ppf(conf)
    if (m is not None): return m, m-h, m+h
    else: return h

# compute confidence intervals for parameters
for ss in params.index.values:
	for pp in ['k_log','gr','lagC']:
		ci1 = conf_int(
			conf=confidence,
			std=params.loc['Ancestor','std({})'.format(pp)],
			m=params.loc['Ancestor','mean({})'.format(pp)])
		ci2 = conf_int(
			conf=confidence,
			std=params.loc[ss,'std({})'.format(pp)],
			m=params.loc[ss,'mean({})'.format(pp)])
		params.loc[ss,'sig({})'.format(pp)] = detSigDiff(ci1[1:],ci2[1:])


###############################################
# Prep Figure, Plotting Functions and Globals #
##############################################

# initialize figure
fig = plt.figure(constrained_layout=False,figsize=[16.5,16])
spec = gridspec.GridSpec(nrows=4,ncols=4,height_ratios=[2.25,.15,1,1],figure=fig)

ax00 = fig.add_subplot(spec[0,0]) # sum of functional differences
ax01 = fig.add_subplot(spec[0,1]) # carrying capacity
ax02 = fig.add_subplot(spec[0,2]) # auc
ax03 = fig.add_subplot(spec[0,3])#,sharey=ax00) # lag time

# set global aesthetic parameters
def prettify_bar_plot(ax,fontsize=20):
	ax.yaxis.grid(False)
	[spine.set(lw=2,color='black') for _,spine in ax.spines.items()]
	[ii.set(fontsize=fontsize) for ii in ax.get_xticklabels()+ax.get_yticklabels()]

def prettify_line_plot(ax,fontsize=20):
	ax.xaxis.grid(False)
	[spine.set(lw=2,color='black') for _,spine in ax.spines.items()]
	[ii.set(fontsize=fontsize) for ii in ax.get_xticklabels()+ax.get_yticklabels()]

def integerize_ticks(ax):
	[ii.set_major_locator(MaxNLocator(integer=True)) for ii in [ax.xaxis]]

fontsize = 20 

###############################
# Plot Functional Differences #
###############################

ax = ax00

kwargs = {'height':0.7,
          'color':'#1f78b4',
          'alpha':0.4} 

# define mean and bands (x-axis)
xvalues = funcdiffs.Func_Diff_Mean
xbands = [eval(ii)[1]-jj for ii,jj in zip(funcdiffs['Func_Diff_CI'],funcdiffs['Func_Diff_Mean'])]

# define y-axis positions and tick labels and their positions
ypos = range(0,funcdiffs.shape[0])
ypos_ticks = [ii for ii in ypos]
ypos_labels = funcdiffs.Line.values
plt.setp(ax,yticks=ypos_ticks,yticklabels=ypos_labels)

# plot bars
ax.barh(ypos,xvalues,xerr=xbands,**kwargs)
ax.set_xlim([0,np.ceil(ax.get_xlim()[1]+1)])
ax.set_title(r'$\Vert OD\Delta\Vert$',fontsize=fontsize,y=1.05)
prettify_bar_plot(ax,fontsize)
integerize_ticks(ax)

###################
# Plot Parameters #
###################

titles_dict = {'mean(k_log)':'Carrying Capacity',
               'mean(gr)':'Growth Rate',
               'mean(lagC)':'Lag Time (Hours)'}

xlim_dict = {'mean(k_log)':[0,1.2],
             'mean(gr)':[0,0.04],
             'mean(lagC)':[0,24]}

# use ypos, ypos_ticks, and ypos_labels from earlier
for ax,param in zip([ax01,ax02,ax03],['k_log','gr','lagC']):

	# pointers to parameters
	mu_param = 'mean({})'.format(param)
	sig_param = 'sig({})'.format(param)

	# define mean and bands (x-axis)
	x_values = list(params.loc[ypos_labels,mu_param].values)
	xbands = params.loc[ypos_labels,mu_param.replace('mean','std')].values
	xbands = [ii*norm.ppf(confidence) for ii in xbands]

	# plot bars
	ax.barh(ypos,x_values,xerr=xbands,**kwargs)  
	ax.set_title(titles_dict[mu_param],fontsize=fontsize,y=1.05)
	prettify_bar_plot(ax,fontsize)

	# adjust x-axis limits and labels
	x0,x1=xlim_dict[mu_param]
	ax.set_xlim([x0,x1])
	plt.setp(ax,yticklabels=[])
	plt.setp(ax,xticks=np.linspace(x0,x1,5))

	# determine significance astersiks and plot
	sig_values = list(params.loc[ypos_labels,sig_param].values)
	sig_values = ['*' if ii else '' for ii in sig_values]
	for ypos_tick,sig_value in zip(ypos_ticks,sig_values):
		ypos_ast = ypos_tick-kwargs['height']*.3
		ax.text(x=0.875*x1,y=ypos_ast,s=sig_value,transform=ax.transData,
			    fontsize=fontsize*1.5,ha='center',va='center')	  

###############
# Plot Curves #
###############

for ii,strain in enumerate(dunphy_order_strains[1:][::-1]):
	ax_x = 2+int(ii/4)
	ax_y = ii - 4*(ax_x-2)
	ax = fig.add_subplot(spec[ax_x,ax_y])
	ax.text(0.5,0.9,strain,transform=ax.transAxes,
		    fontsize=fontsize,ha='center',va='center')

	d0 = getMeanBands(gp_data[gp_data.Strain=='Ancestor'])
	ax.plot(d0['x'],d0['y'],color=(0,0,0,0.8),lw=3)
	ax.fill_between(d0['x'],d0['y1'],d0['y2'],color=(0,0,0,0.1))

	d1 = getMeanBands(gp_data[gp_data.Strain==strain])
	ax.plot(d1['x'],d1['y'],color=(0,0,1,0.8),lw=3)
	ax.fill_between(d1['x'],d1['y1'],d1['y2'],color=(0,0,1,0.1))
	
	prettify_line_plot(ax,fontsize)

	if ax_y == 0: ax.set_ylabel('ln OD',fontsize=fontsize)
	if ax_x == 3: ax.set_xlabel('Time (Hours)',fontsize=fontsize)
	if ax_y > 0: plt.setp(ax,yticklabels=[])

	ax.set_xlim([0,48])
	ax.set_ylim([0,1])

	plt.setp(ax,xticks=np.linspace(0,48,5));  


plt.subplots_adjust(hspace=0.3,wspace=0.25)
filename='Midani_AMiGA_Supp_Figure_7'
plt.savefig('./figures/{}.pdf'.format(filename),bbox_inches='tight')


