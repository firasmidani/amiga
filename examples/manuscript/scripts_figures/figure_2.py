#!/usr/bin/env python

# __author__ = Firas S Midani
# __email__ = midani@bcm.edu

# Midani et al. (2020) Figure 2 is generated by this script 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')

def read_csv(foo): return pd.read_csv(foo,sep='\t',header=0,index_col=0)

# read model estimated parameters
summ_df ='./biolog/summary/pooled_by_isolate_summary_normalized.txt'    
summ_df = read_csv(summ_df)

# replace 'negative control'  with 'no carbon control' to avoid ambiguity
summ_df = summ_df.replace('Negative Control','No-carbon Control')

# read model predictions
split_df ='./biolog/derived/pooled_by_isolate_gp_data.txt'
poold_df = './biolog/derived/split_gp_data.txt'
    
split = read_csv(split_df)
poold = read_csv(poold_df)

# reduce dat to CD2015 on PM1
summ_df = summ_df[summ_df.isin({'Isolate':['CD2015'],'PM':[1]}).sum(1)==2]
poold = poold[poold.Plate_ID=='CD2015_PM1-1']

# get substrates that support normalized K > 1.2 and include no-carbon control
substrates = summ_df[summ_df.loc[:,'norm(auc_log)']>1.2].Substrate.values
tmp_summ = summ_df[summ_df.Substrate.isin(list(substrates)+['No-carbon Control'])]

# reduce data to substrates
sample_ids = tmp_summ.index
tmp_data = poold[poold.Sample_ID.isin(sample_ids.values)]
tmp_data = tmp_data.pivot(index='Time',columns='Sample_ID',values='GP_Output')

# assign color
def assignGroupColor(row):
    if row['diauxie'] == 1: return (0.85,0.37,0.01)
    elif row['norm(gr)'] == 1.0: return (0,0,0,0.8)
    elif row['norm(gr)'] > 1.20: return (0.11,0.62,0.47)
    elif row['norm(gr)'] < 1.20: return (0.46,0.44,0.70)

colors = pd.DataFrame(tmp_summ.apply(lambda x: assignGroupColor(x), axis=1),columns=['color'])
tmp_summ = tmp_summ.join(colors)

# assign label location
tmp_locs = tmp_data.iloc[-1,:].sort_values()
tmp_locs = pd.DataFrame(data=np.linspace(0.05,0.95,tmp_locs.shape[0]),
                        columns=['label'],index=tmp_locs.index,)
tmp_summ = tmp_summ.join(tmp_locs)

# plot
def largeTickLabels(ax): [ii.set(fontsize=20) for ii in ax.get_xticklabels()+ax.get_yticklabels()]

fig, (ax0,ax1) = plt.subplots(1,2,figsize=[18,6],sharey=False)

sub_time = tmp_data.index
ymin,ymax = ax0.get_ylim()

ax1_y = 'norm(gr)'
ax1_x = 'norm(auc_log)'

for idx,row in tmp_summ.iterrows():
    sub,color,label = row['Substrate'],row['color'],row['label']
    ax0.plot(tmp_data.index,tmp_data.loc[:,idx].values,color=color,lw=4,alpha=0.65)
    ax0.text(1.04,label,sub,fontsize=20,fontweight='normal',va='center',color=color,
            transform=ax0.transAxes)
    ax1.scatter(row[ax1_x],row[ax1_y],s=500,color=color,edgecolor='white',alpha=0.8,zorder=5)

ax0.grid(False)
ax1.grid(False)

[ax0.spines[ii].set(lw=0) for ii in ['top','bottom','right']]
[ax1.spines[ii].set(lw=0) for ii in ['top','bottom','right','left']]

largeTickLabels(ax0)
largeTickLabels(ax1)

ax0.axhline(0,0,1,lw=2,color='k')
ax1.axhline(1,0,1,lw=2,color=(0,0,0,1),zorder=2)
ax1.axvline(1,0,1,lw=2,color=(0,0,0,1),zorder=2)

ax0.spines['left'].set(lw=2,color=(0,0,0,0.8))
#ax1.spines['left'].set(lw=2,color=(0,0,0,0.8))

ax0.set_xlabel('Time (hours)',fontsize=20)
ax0.set_ylabel('ln OD',fontsize=20)

ax1.set_xlabel('     Normalized Area Under the Curve',fontsize=20)
ax1.set_ylabel('Normalized Growth Rate',fontsize=20)

ax0.set_xlim([-0.5,22.5])
ax0.set_ylim([-0.05,2.25])

ax1.set_xlim([.94,1.7])
ax1.set_ylim([.84,1.7])

plt.setp(ax0,xticks=np.linspace(0,20,5))
plt.setp(ax0,yticks=np.linspace(0,2,3))

plt.setp(ax1,xticks=np.linspace(1,1.6,4))
plt.setp(ax1,yticks=np.linspace(1,1.6,4))

ax1.axhline(1.2,0,1,color=(0.5,0.5,0.5,.5),zorder=2,lw=2,ls='--')
ax1.axvline(1.2,0,1,color=(0.5,0.5,0.5,.5),zorder=2,lw=2,ls='--')

# adjust spacing between panels
plt.subplots_adjust(wspace=1.0)

# add panel letters
ax0.text(transform=ax0.transAxes,x=-0.225,y=1,ha='left',va='top',s='A',fontsize=30,fontweight='bold')
ax1.text(transform=ax1.transAxes,x=-0.225,y=1,ha='left',va='top',s='B',fontsize=30,fontweight='bold')

filename='Midani_AMiGA_Figure_2'
plt.savefig('./figures/{}.pdf'.format(filename),bbox_inches='tight')
