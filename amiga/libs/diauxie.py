#!/usr/bin/env python

'''
AMiGA library for diauxic shift detection and characterization.
'''

__author__ = "Firas Said Midani"
__email__ = "midani@bcm.edu"


# TABLE OF CONTENTS (3 functions)

# detectDiauxie
# pad
# mergePhases

import numpy as np # type: ignore
import pandas as pd # type: ignore

from .utils import getValue


def detectDiauxie(x,y0,y1,y2,cov0,cov1,thresh,varb='K'):
    '''
    Decompose a growth curve into individual growth phases separated by OD inflection.
    
    Args:
        x (numpy.ndarray): time
        y0 (numpy.ndarray): mean of latent function
        y1 (numpy.ndarray): mean of first derivative of latent function (i.e. growth rate)
        y2 (numpy.ndarray): mean of second dderivative of latent function (e.g. acceleration)
        cov0 (numpy.ndarray): covariance of latent function
        cov1 (numpy.ndarray): covariance of first derivatie of latent function
        thresh (float): ?
        varb (str): use either 'K' or 'r' to threshold/call secondary growth curves
    
    Retuns:
        ret (pandas.DataFrame): dataframe summarizes each growth phase with following:
            t_left: time at left bound
            t_right: time at right bound
            K: total growth
            r: maximum growth rate
            r_left: growth rate at left bound
            r_right: growth rate at ight bound
    '''

    if varb == 'K':
        second_varb = 'r'
    else:
        second_varb = 'K'

    if x.ndim>1:
        x = x[:,0].ravel() # assumes time is first dimension

    # unravel these numpy.ndarray arguments (avoids issues in line 97) 
    # <-- this may cause problems if x is multi-dimensional [NEED TO VERIFY] -->
    y0, y1, y2 = (ii.ravel() for ii in [y0, y1, y2])

    # indices for inflections
    #ips = list(np.where(np.diff(np.sign(y2.ravel())))[0])
    ips = list(np.where(np.diff(np.sign(y2)))[0])

    cond_1 = len(ips)==0
    cond_2 = (len(ips)==1 and ips[0]==0)
    cond_3 = (len(ips)==1 and ips[0]==len(y2)-1)
    cond_4 = np.max(y0) < getValue('diauxie_k_min')
    
    if cond_1 or cond_2 or cond_3 or cond_4:
        cols = ['t_left','t_right','K','r','r_left','r_right']
        ret = pd.DataFrame([x[0],x[-1],np.max(x),np.max(y1),y1[0],y1[-1]],index=cols)
        return ret.T
    
    # types of inflections
    its = [np.sign(y2[ii+1]) if ii<(len(y2)-2) else -1*np.sign(y2[ii-1]) for ii in ips]

    # pad edge cases 
    ips,its = pad(ips,its,edge=1,length=len(y2))
    ips,its = pad(ips,its,edge=-1,length=len(y2))
    
    # convert data types
    ips = np.array([int(ii) for ii in ips])
    its = np.array(its)

    # define bounds of each growth stage
    starts = np.where(its==1)[0][:-1]
    stops = starts+2

    # initialize a summary dataframe and populate with bounds
    #ret = np.zeros((int(len(ips)/2),7))
    ret = np.zeros((len(starts),7))
    ret[:,0] = ips[starts]
    ret[:,1] = ips[stops]
    
    # compute several metrics for growth stage (should I use absolute?)
    bounds = [(int(ii[0]+1),int(ii[1]+1)) for ii in ret]
    ret[:,2] = [np.max(y0[left:right]-y0[left]) for left,right in bounds] # Total change in OD
    ret[:,3] = [np.max(y1[left:right]) for left,right in bounds]         # max growth rate, 
    ret[:,[4,5]] = [[y1[left-1],y1[right-1]] for left,right in bounds]     # growth rate at both bounds 
    # define attraction of each growth stage: a growth stage is attrached 
    #   to the adjacent gowth stage with the least difference in terms of 
    #   growth rate at the shared bounds (relative to max growth rate 
    #   within the bounds)  
    ret[:,6] = [-1 if np.abs(row[5]-row[3]) > np.abs(row[4] - row[3]) else 1 for row in ret]

    # annotate datafame and sort in ascending order
    cols = ['t_left','t_right','K','r','r_left','r_right','attraction']
    #cols = ['ind0','ind1','y_delta','max_y1','y1(ind0)','y1(ind1)','attraction']
    ret = pd.DataFrame(ret,columns=cols)

    # how to deal with negative r or K
    #   if at least one value is nonzero positive
    if any(ii>0 for ii in ret[varb].values):
        # starting with the smallest growth stage (smallest total change in OD):
        #    if it's K is smaller than a certain proportion of the max K
        #    merge with attractor, continue until all growth phases meet criteria
        while ret[varb].min() < thresh*ret[varb].max() :
            
            ret = ret.sort_values(['t_left'])
            ret.iloc[0,-1] = 1   # first phase is always attracted forward in time
            ret.iloc[-1,-1] = -1 # last phase is always attracted backward in time
            
            ret = ret.sort_values([varb,second_varb])
            idx = ret.index.values[0]
            att = ret.loc[idx,'attraction']
            att = idx+att
            ret = mergePhases(ret,idx,att,varb=varb)
            
            # should you re-compute attraction?
    else:  
        while ret.shape[0] > 1:  # coalescale all into a single curve
            ret = mergePhases(ret,0,1)

    # re-sort by time and convert array indices to time values
    ret = ret.sort_values(['t_left'])
    ret.iloc[:,0] = ret.iloc[:,0].apply(lambda i: x[int(i)])
    ret.iloc[:,1] = ret.iloc[:,1].apply(lambda i: x[int(i)])
    ret.drop('attraction',axis=1,inplace=True)
    
    return ret


def pad(ips,its,edge=1,length=None):
    '''
    pad() is used only within diauxie(). Diauxie functions best if the growth curve
        begins at time zero with a positive inflection and end at last time point
        with another positive inflection. pad() pads the edges of inflection points 
        (ips in time units) and inflection types (its, whether +ve or -ve) with dummy
        inflection points to assist diauxie in analyzing a gowth curve. 
    '''

    # if negative, pad the right edge of the curve, otherwise pad the left
    if edge==-1:
        ips = ips[::-1]
        its = its[::-1]
        idx = length-1
    else:
        idx = 0

    if (ips[0] == 0) and (its[0] == 1):
        pass
    elif (its[0] == -1):
        ips = [idx] + ips
        its = [1] + its
    elif (its[0] == 1):
        ips = [idx,idx] + ips
        its = [1,-1] + its

    # reset the order of the input before retuning
    if edge==-1:
        ips = ips[::-1]
        its = its[::-1]

    return ips,its

def mergePhases(df,row1_idx,row2_idx,varb='K'):
    '''
    mergePhasese() is used only within diauxie(). Diauxie() iterates through
        growth phases (described in df) and decides whether two growth phases
        should be merged and considered as a single phase. mergePhases()
        goes though each attibute (df column) and identifies the final value,
        it drops the rejected growth phase and updates the accepted growth phase. 
        
    Args:
        df (pandas.DataFrame): dataframe summarizes each growth phase with following:
            t_left: time at left bound
            t_right: time at right bound
            K: total growth
            r: maximum growth rate
            r_left: growth rate at left bound
            r_right: growth rate at ight bound 
        row1_idx: index value for rejected growth phase
        row2_idx: index value for accepted growth phase
    
    Return:
        df (pandas.DataFrame): # of rows should be one less than input datafame # of rows
    '''
    
    tmp = df.loc[[row1_idx,row2_idx],:].sort_index()
    tmp = tmp.sort_values(['t_left'])
    
    values = ([tmp['t_left'].min(numeric_only=True),
               tmp['t_right'].max(numeric_only=True),
               tmp['K'].sum(numeric_only=True),
               tmp['r'].max(numeric_only=True),
               tmp['r_left'].values[0],
               tmp['r_right'].values[1],
               tmp.loc[row2_idx,'attraction']])
    
    df.loc[row2_idx,:] = values
    df.drop(row1_idx,axis=0,inplace=True)
    df = df.sort_values(['t_left']).reset_index(drop=True) 
    
    return df
