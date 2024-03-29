#!/usr/bin/env python

# __author__ = Firas S Midani
# __email__ = midani@bcm.edu

# Midani et al. (2020) Supplemntal Figure 5 is generated by this script 

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import norm

def read_csv(foo):
    return pd.read_csv(foo,sep='\t',header=0,index_col=0)

def largeTickLabels(ax,fontsize=20):
    [ii.set(fontsize=fontsize) for ii in ax.get_xticklabels()+ax.get_yticklabels()]
    
def subsetDf(df,criteria):
    
    for k,v in criteria.items():
        if not isinstance(v,list): criteria[k] = [v]
    
    return df[df.isin(criteria).sum(1)==len(criteria)]

# read model predictions

parent ="./death"

df_split = read_csv('{}/derived/split_gp_data.txt'.format(parent))
df_split_summ = read_csv('{}/summary/split_summary.txt'.format(parent))

df_data = read_csv('{}/derived/pooled_gp_data.txt'.format(parent))
df_data_fixed = read_csv('{}/derived/pooled_fixed_noise_gp_data.txt'.format(parent))

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
    
    confidence = .95
    alpha = 1 - confidence
    z_value = 1-alpha/2
    scaler = norm.ppf(z_value)
    
    low = mu - scaler*np.sqrt(Sigma)
    upp = mu + scaler*np.sqrt(Sigma) 
    
    return time,mu,low,upp

def plotModel(ax,df_poold,df_split,criteria,latent=True,bands=False,raw=False,noise=False):
    
    tmp = subsetDf(df_poold,criteria)
    
    xy0 = tmp[tmp.Concentration=='Low']
    xy1 = tmp[tmp.Concentration=='High']

    time0,mu0,low0,upp0 = getLatentFunction(xy0,order=0,add_noise=noise)
    time1,mu1,low1,upp1 = getLatentFunction(xy1,order=0,add_noise=noise)

    ax.plot(time0,mu0,color=color_low,lw=2,label='Low')
    ax.plot(time1,mu1,color=color_high,lw=2,label='High')
    
    if bands:
        ax.fill_between(time0,low0,upp0,color=color_low,alpha=0.1)
        ax.fill_between(time0,low1,upp1,color=color_high,alpha=0.1)
    
    ax.legend(fontsize=15,loc='lower right',frameon=False)

    if raw:
        
        cc = {}
        cc['Low'] = {'Ribotype':criteria['Ribotype'],
                     'Substrate':criteria['Substrate'],
                     'Concentration':['Low']}
        cc['High'] = {'Ribotype':criteria['Ribotype'],
                      'Substrate':criteria['Substrate'],
                      'Concentration':['High']}

        for conc,color in zip(['Low','High'],['green','purple']):
            tmp = subsetDf(df_split_summ,cc[conc]).index.values
            tmp = subsetDf(df_split,{'Sample_ID':list(tmp)})
            tmp = tmp.pivot(index='Time',values='GP_Input',columns='Sample_ID')
            ax.plot(tmp.index.values,tmp.values,color=color,lw=0.5,alpha=0.5)

fig,axes_all = plt.subplots(4,2,figsize=[10,16],sharex=False,sharey=True)

fontsize = 20
color_low = '#4daf4a'
color_high = '#984ea3'
color_diff = '#377eb8'

# 1st row: raw curves with latent function

# 2nd row: latent function and gaussian noise
#          no measurement noise added to confidence interval

# 3rd row: latent function and gaussian noise
#          measurement noise added to confidence interval

# 4th row: latent function and gaussian noise
#          empirically estimated measurement noise added to confidence interval

criteria = {'Ribotype':['RT027'],'Substrate':['Fructose']}

ax = axes_all[0,0]
plotModel(ax,df_data,df_split,criteria,latent=True,bands=False,raw=True)
ax.set_title('RT027 on Fructose',fontsize=fontsize)

ax = axes_all[1,0]
plotModel(ax,df_data,df_split,criteria,latent=True,bands=True,raw=False,noise=False)

ax = axes_all[2,0]
plotModel(ax,df_data,df_split,criteria,latent=True,bands=True,raw=False,noise=True)

ax = axes_all[3,0]
plotModel(ax,df_data_fixed,df_split,criteria,latent=True,bands=True,raw=False,noise=True)

criteria = {'Ribotype':['RT053'],'Substrate':['Fructose']}

ax = axes_all[0,1]
plotModel(ax,df_data,df_split,criteria,latent=True,bands=False,raw=True)
ax.set_title('RT053 on Fructose',fontsize=fontsize)

ax = axes_all[1,1]
plotModel(ax,df_data,df_split,criteria,latent=True,bands=True,raw=False,noise=False)

ax = axes_all[2,1]
plotModel(ax,df_data,df_split,criteria,latent=True,bands=True,raw=False,noise=True)

ax = axes_all[3,1]
plotModel(ax,df_data_fixed,df_split,criteria,latent=True,bands=True,raw=False,noise=True)


for ax in np.ravel(axes_all):
    ax.set_ylim([-0.35,2.35])
    plt.setp(ax,xticks=np.linspace(0,24,5))
    plt.setp(ax,yticks=np.linspace(0,2,5))
    largeTickLabels(ax,fontsize)

for ax in axes_all[:,0]:
    ax.set_ylabel('ln OD',fontsize=fontsize)
    
for ax in axes_all[-1,:]:
    ax.set_xlabel('Time (hours)',fontsize=fontsize)
    
# labels = [(0.888,'A'),
#           (0.688,'B'),
#           (0.489,'C'),
#           (0.288,'D')]

# for (pos,lab) in labels:
#     plt.text(0.03,pos,lab,ha='center',va='center',fontsize=20,fontweight='normal',
#              transform=fig.transFigure)   
    
# adjust spacing between panels
plt.subplots_adjust(hspace=0.3)

# add panel axes_all
axes_all[0,0].text(transform=axes_all[0,0].transAxes,x=-0.275,y=1,ha='left',va='top',s='A',fontsize=30,fontweight='bold')
axes_all[1,0].text(transform=axes_all[1,0].transAxes,x=-0.275,y=1,ha='left',va='top',s='B',fontsize=30,fontweight='bold')
axes_all[2,0].text(transform=axes_all[2,0].transAxes,x=-0.275,y=1,ha='left',va='top',s='C',fontsize=30,fontweight='bold')
axes_all[3,0].text(transform=axes_all[3,0].transAxes,x=-0.275,y=1,ha='left',va='top',s='D',fontsize=30,fontweight='bold')

# save figure as PDF
filename = 'Midani_AMiGA_Supp_Figure_6'
plt.savefig('./figures/{}.pdf'.format(filename),bbox_inches='tight')