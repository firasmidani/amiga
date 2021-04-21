#!/usr/bin/env python

# __author__ = Firas S Midani
# __email__ = midani@bcm.edu

# Midani et al. (2020) Supplemntal Figure 3 is generated by this script 

import os
import GPy
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import norm

sns.set_style('whitegrid')

def read_csv(foo): return pd.read_csv(foo,sep='\t',header=0,index_col=0)

# read model prediction

summ_df ='./biolog/summary/pooled_by_isolate_summary_normalized.txt'
poold_df = './biolog/derived/pooled_by_isolate_gp_data.txt'

df_summ = read_csv(summ_df)
df_data = read_csv(poold_df)

# plotting functions

def subsetDf(df,criteria):
    
    for k,v in criteria.items():
        if not isinstance(v,list): criteria[k] = [v]
    
    return df[df.isin(criteria).sum(1)==len(criteria)]

def getLatentFunction(df,order=0,add_noise=False):
    '''
    Get arrays for time, mean, lower and upper bounds of confidence interval. 

    Args:
    	df (pandas.DataFrame): in the format of gp_data from AMiGA, so columns must
    		include 'mu','mu1','Sigma','Sigma1','Noise'.
    	order (int): choose zero-order (0) or fist-order derivative (1).
    	add_noise (boolean): whether to include estimated noise to confidence intervals.
    '''

    time = df.Time.values
    
    if order==0:
        mu = df.mu.values
        Sigma = df.Sigma.values
        if add_noise: Sigma = Sigma + df.Noise.values
    else:
        mu = df.mu1.values
        Sigma = df.Sigma1.values
            
    scaler = norm.ppf(0.95)
    
    low = mu - scaler*np.sqrt(Sigma)
    upp = mu + scaler*np.sqrt(Sigma) 
    
    return time,mu,low,upp

def largeTickLabels(ax): [ii.set(fontsize=20) for ii in ax.get_xticklabels()+ax.get_yticklabels()]


def plot_fit_full(df,fig_ax=None):
    
    if fig_ax is None: fig,ax = plt.subplots(3,1,figsize=[5,15],sharex=True);
    else: fig,ax = fig_ax
        
    df = df.sort_values(['Time'])
    
    # plot latent function
    x,y,ymin,ymax = getLatentFunction(df,order=0)
    ax[0].plot(x,y,color=(0,0,0,0.9),lw=4)
    y0 = y.ravel()
    
    # plot derivative of latent function
    x,y,ymin,ymax = getLatentFunction(df,order=1)
    ax[1].plot(x,y,color=(0,0,0,0.9),lw=4)
    y1 = y.ravel()
    
    m = GPy.models.GPRegression(x[:,np.newaxis],y[:,np.newaxis])
    m.optimize()
    mu,_ = m.predictive_gradients(x[:,np.newaxis])
    y2 = mu[:,0].ravel()
    
    ax[2].plot(x,y2,color=(0,0,0,0.9),lw=4)
    
    kwargs0 = {'xlabel':'', 'ylabel':'ln OD(t)/OD(0)'}
    kwargs1 = {'xlabel':'Time (hours)','ylabel':'d/dt [ln OD(t)/OD(0)]'}
    plt.setp(ax[0],**kwargs0);
    plt.setp(ax[1],**kwargs1);
    

    return fig,ax,x[:,np.newaxis],y0,y1,y2

# initialize figure
fontsize=20
criteria = {'Substrate':'D-Sorbitol',  'Isolate':'CD2015',  'PM':1}

sub_data = subsetDf(df_data,criteria)

_,ax,x,y0,y1,y2 = plot_fit_full(sub_data);

t0 = ax[0].set_ylabel(r'$ln\:OD(t)$',fontsize=20)
t1 = ax[1].set_ylabel(r'$\frac{d}{{dt}}\:ln\:OD(t)$',fontsize=20)
t2 = ax[2].set_ylabel(r'$\frac{d^2}{{dt}^2}\:ln\:OD(t)$',fontsize=20)

# adjust window limites
ax[0].set_xlim([0,24])
ax[0].set_ylim([-0.2,2.3])
ax[1].set_ylim([-0.2,0.4])
ax[2].set_ylim([-0.2,0.2])

# adjust window tick labels and label sizes
plt.setp(ax[0],xticks=np.linspace(0,24,5))
plt.setp(ax[0],yticks=np.linspace(0,2,5))
plt.setp(ax[1],yticks=np.linspace(-0.2,0.4,4))
plt.setp(ax[2],yticks=np.linspace(-0.2,0.2,5))

[largeTickLabels(ax_ii) for ax_ii in ax];

# adjust axes labels
ax[1].set_xlabel('')
ax[2].set_xlabel('Time (hours)',fontsize=fontsize)

# identify phase dividers and anntoate

shifts = []
annotations = []
count = 1;

ymin,ymax = ax[0].get_ylim()
xmin,xmax = ax[0].get_xlim()
sign_diff = np.where(np.diff(np.sign(y2.ravel())))[0] ## all inflection points

for sd in list(sign_diff):
    
    # add red scatter point for inflection
    ax[2].scatter(x[sd][0],0,s=100,marker='o',color=(1,0,0,0.8),zorder=10)
    
    # if positive inflection, characteize
    if (y2[sd+1]-y2[sd-1])>0: # if positivee inflection point 
        x_time = x[sd][0]
        x_pos = (x_time - xmin) / (xmax - xmin)
        x_str = 'P{} '.format(count)
        count += 1
        annotations.append((x_time,x_pos,x_str))

# handle the inflection point at the end of the growth cuve
if sd != len(x):
    x_time = x[-1]
    x_pos = (x_time - xmin) / (xmax - xmin)
    x_str = 'P{} '.format(count)
    annotations.append((x_time,x_pos,x_str))
    
# adjust figure spines and and add phase dividers
for ax_ii in ax: 
    
    [ax_ii.spines[ii].set(lw=0) for ii in ['top','bottom','right']];
    ax_ii.spines['left'].set(lw=2,color=(0,0,0,0.8))
    ax_ii.axhline(0,0,1,lw=2,color=(0,0,0,0.8))
    ax_ii.grid(False)

    for x_time,x_pos,x_str in annotations:
        ax_ii.axvline(x_time,0,.98,lw=2,ls='--',color=(0,0,0,0.8))
        ax_ii.text(transform=ax_ii.transAxes,s=x_str,x=x_pos,y=.98,va='top',ha='right',fontsize=20)
    
# add panel letters
ax[0].text(transform=ax[0].transAxes,x=-0.29,y=1,ha='left',va='top',s='A',fontsize=30,fontweight='bold')
ax[1].text(transform=ax[1].transAxes,x=-0.29,y=1,ha='left',va='top',s='B',fontsize=30,fontweight='bold')
ax[2].text(transform=ax[2].transAxes,x=-0.29,y=1,ha='left',va='top',s='C',fontsize=30,fontweight='bold')

# save figure as PDF and convert to EPS
filename = 'Midani_AMiGA_Supp_Figure_1'
plt.savefig('./figures/{}.pdf'.format(filename),bbox_inches='tight')
