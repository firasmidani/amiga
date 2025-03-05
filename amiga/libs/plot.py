#!/usr/bin/env python

'''
AMiGA library of auxiliary functions for plotting suppport.
'''

__author__ = "Firas S Midani"
__email__ = "midani@bcm.edu"

# TABLE OF CONTENTS (7 functions)

# largeTickLabels
# savePlotWithLegends
# setAxesLabels
# dynamicWindowAdjustment
# addRealPlotLine
# addMVNPlotLines
# plotDeltaOD

import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore

from scipy.stats import norm # type: ignore

from .utils import subsetDf, getValue, getTimeUnits


def largeTickLabels(ax,fontsize=20):

    [ii.set(fontsize=fontsize) for ii in ax.get_xticklabels()+ax.get_yticklabels()]

    return ax


def savePlotWithLegends(ax,fig_path,loc,fontsize=20):
    '''
    Given an ax (matplotlib.axes._subplots.AxesSubplot) and path (str), save the figure
        with legend outside on the center left and keep a tight box around contents. 
    '''

    handles, labels = ax.get_legend_handles_labels()
    if loc == 'outside':
        lgd = ax.legend(handles,labels,loc='center left',bbox_to_anchor=(1,0.5),fontsize=fontsize)
    elif loc == 'inside':
        lgd = ax.legend(handles,labels,loc='lower left',fontsize=fontsize)
    plt.savefig(fig_path,bbox_extra_artists=(lgd,), bbox_inches='tight')


def setAxesLabels(ax,subtract_control,plot_params,logged=True,fontsize=20):
    ''''
    Given an axis and analysis parameters, determine appropriate labels 
        for axes and adjus them accordingly. 

    Args:
        ax (matplotlib.axes._subplots.AxesSubplot) 
        subtract_control (boolean)
        plot_params (dictionary)
        fontsize (float)

    Returns:
        ax (matplotlib.axes._subplots.AxesSubplot) 
    '''
    import matplotlib as mpl
    mpl.rcParams["mathtext.default"] = 'regular'
    mpl.rcParams["font.family"] = 'sans-serif'
    mpl.rcParams["font.sans-serif"] = 'Arial'
    # mpl.rcParams["text.usetex"] = True

    #if plot_params['plot_linear_od']:
    #    base = getValue('hypo_plot_y_label')
    #    base = r'$\frac{{{}}}{{{}}}$'.format(base+'(t)',base+'(0)')
    #else:
    if logged:
        base = 'ln {}'.format(getValue('hypo_plot_y_label'))
    else:
        base = getValue('hypo_plot_y_label')

    # plot aesthetics
    if subtract_control:
        ylabel = f'Normalized {base}'
    else:
        ylabel = base

    ax.set_xlabel('Time ({})'.format(getTimeUnits('output')),fontsize=plot_params['fontsize'])
    ax.set_ylabel(ylabel,fontsize=plot_params['fontsize'])

    return ax


def dynamicWindowAdjustment(ax):
    '''
    Adjusts the y-axis limits for figure assembled in plotHypothesis. For the to_permute
        panels, adjust all axes to accomodate most minimal and most maximal point across
        all sub-plots in the row. For the bottow panel, adjust all axes to accomodate the
        maximum absolute point while also maintaining symmetry so that the y-axis is 
        centered at zero.

    Args:
        ax (matplotlib.axes._subplots.AxesSubplot)

    Returns:
        ax (matplotlib.axes._subplots.AxesSubplot)
    '''

    def adjustWindow(ax,ymin,ymax,symmetric=False):
        '''
        Given an axis, and y-limits (ymin,ymax), and symmetry choice, adjust y-axis.
        '''
        if symmetric:
            ax.set_ylim([-1*np.ceil(ymin),np.ceil(ymax)])
        else:
            ax.set_ylim([np.floor(ymin),1*np.ceil(ymax)])
        return ax

    if isinstance(ax,np.ndarray):

        if len(ax.shape) > 1: ## conditioning

            # track min and max of plot axis limits
            y_min,y_max = 0,0
            for ax_ii in ax[0,:]:
                y_min = np.min([y_min,ax_ii.get_ylim()[0]])
                y_max = np.max([y_max,ax_ii.get_ylim()[1]])

            # adjust all axes to min and max
            for ax_ii in ax[0,:]:
                ax_ii = adjustWindow(ax_ii,y_min,y_max,symmetric=False)

            # track max of plot axis absolute limits
            y_max = 0
            for ax_ii in ax[1,:]:
                y_max = np.max([y_max,np.abs(ax_ii.get_ylim()[0]),np.abs(ax_ii.get_ylim()[1])])

            # adjust all axes to min and max
            for ax_ii in ax[1,:]:
                ax_ii = adjustWindow(ax_ii,y_max,y_max,symmetric=True)

        elif len(ax.shape) == 1: ## no conditioning
        
            # round window limits
            y_min,y_max = ax[0].get_ylim()
            ax[0] = adjustWindow(ax[0],y_min,y_max)

            # symmetrical and rounded window limits
            y_max = np.max([np.abs(ax[1].get_ylim()[0]),np.abs(ax[1].get_ylim()[1])])
            ax[1] = adjustWindow(ax[1],y_max,y_max,symmetric=True)

    else:
        # round window limits
        y_min,y_max = ax.get_ylim()
        ax = adjustWindow(ax,y_min,y_max)

    return ax


def addMVNPlotLine(ax,x,criteria,label,z_value,color,plot_params,noise=False):

    '''
    Given data (x) and criteria, find relevant sample IDs and plot them on axis.

    Args:
        ax (matplotlib.axes._subplots.AxesSubplot)
        x (pandas.DataFrame): must include columns for Time, mu, Sigma
        criteria (dictionary): keys must be column headers in x, values must be values in x.
        label (str): used for legend label of plotted line.
        z_value (float): z-value for computing confidence interval
        color (str or (R,G,B,A)) where R,G,B,A are floats [0,1]
        plot_params (dictionary)
        noise (boolean): whetehr to plot 95-pct credibel intervals including sample uncertainty

    Returns:
        ax (matplotlib.axes._subplots.AxesSubplot)    
    '''
    scaler = norm.ppf(z_value) # define confidence interval scaler for MVN predictions
    x = subsetDf(x,criteria) # grab value-specific model predictions

    if noise:
        Sigma = x.Sigma + x.Noise
    else:
        Sigma = x.Sigma

    # compute credible interval
    xtime = x.Time
    y_avg = x.mu
    y_low = x.mu-scaler*np.sqrt(Sigma)
    y_upp = x.mu+scaler*np.sqrt(Sigma)

    # convert from log2 to linear OD
    # if plot_params['plot_linear_od']:
    #     y_avg = np.exp(y_avg)
    #     y_low = np.exp(y_low)
    #     y_upp = np.exp(y_upp)

    ax.plot(xtime,y_avg,color=color,label=label,alpha=0.9,lw=3.0,zorder=10)
    ax.fill_between(x=xtime,y1=y_low,y2=y_upp,color=color,alpha=0.10,zorder=5)
    ax = largeTickLabels(ax,fontsize=plot_params['fontsize']) 

    #if plot_params['plot_linear_od']:
    #    ax.axhline(y=1,xmin=0,xmax=xtime.max(),lw=3.0,color=(0,0,0,1))
    #else:
    ax.axhline(y=0,xmin=0,xmax=xtime.max(numeric_only=True),lw=3.0,color=(0,0,0,1))

    return ax


def addRealPlotLine(ax,plate,criteria,color,plot_params):
    '''
    Given data (plate) and criteria, find relevant sample IDs and plot them on axis.

    Args:
        ax (matplotlib.axes._subplots.AxesSubplot)
        plate (GrowthPlate object)
        criteria (dictionary): keys must be column headers in plate.key, values must be 
            values in plate.key.
        color (str or (R,G,B,A)) where R,G,B,A are floats [0,1]
        plot_params (dictionary)

    Returns:
        ax (matplotlib.axes._subplots.AxesSubplot)    
    '''

    if plot_params['overlay_actual_data']:

        samples = list(subsetDf(plate.key,criteria).index)

        if len(samples)==0:
            return ax 

        time = plate.time.copy()
        data = plate.data.copy()

        #if plot_params['plot_linear_od']:
        #    data = data.apply(np.exp).copy()

        wide_df = time.join(data)

        wide_df = wide_df.reindex(['Time']+samples,axis=1).set_index('Time')
        wide_df = wide_df.dropna(axis=1) ## necessary to get rid of controls
        ax.plot(wide_df,color=color,alpha=0.5,lw=1,zorder=1)

    return ax


def plotDeltaOD(ax,df,ylabel,xlabel,fontsize=20):
    '''
    Plots Delta OD: its mean (Avg key in df) and credible intervals (Low and Upp keys). 

    Args:
        ax (matplotlib.axes._subplots.AxesSubplot)
        df (pandas.DataFrame)
        ylabel (str)
        xlabel (str)
        fontsize (float)

    Returns: 
        ax (matplotlib.axes._subplots.AxesSubplot)    
    '''

    ax.plot(df.Time,df.Avg,lw=3.0)
    ax.fill_between(df.Time,df.Low,df.Upp,alpha=0.1)
    ax.axhline(y=0,xmin=0,xmax=df.Time.max(numeric_only=True),lw=3.0,color=(0,0,0,1))
    ax = largeTickLabels(ax,fontsize=fontsize)

    if xlabel:
        ax.set_xlabel('Time ({})'.format(getTimeUnits('output')),fontsize=fontsize)
    if ylabel:
        ax.set_ylabel(r'${\Delta}$(ln OD)',fontsize=fontsize)

    return ax 
