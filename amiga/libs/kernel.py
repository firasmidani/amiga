#!/usr/bin/env python

'''
AMiGA library for building kernels for Gaussian Process models.
'''

__author__ = "Firas S Midani"
__email__ = "midani@bcm.edu"


# TABLE OF CONTENTS (3 functions)

# buildKernel
# buildCompositeKernel
# addFixedKernel

import math
import numpy as np # type: ignore
import pandas as pd # type: ignore

from GPy.models import GPRegression # type: ignore
from GPy.kern import RBF,Fixed # type: ignore

from .config import config


def initHyperParams(ndim,x,y):
    '''
    Initializes hyper-parameters for RBF kernel based on method defined in libs/config.py
    Moment-based normalization sets lengthscale initial value to its range (max value - min value)
    and variance initial value is optimized with grid search centered around variance of observations.
    '''

    def oom(number): return math.floor(math.log(number,10))

    method = config['initialization_method']

    if method == 'default':
        return 1,1
    
    if method.startswith('moments-based'):
        
        lengthscale = np.max(x)-np.min(x)

        if method.endswith('fast') and ndim == 1:
            variance = np.var(y)
        elif method.endswith('proper') and ndim > 1:
            variance = np.var(y[:,0])

    if method.endswith('proper'):

        max_oom = 2 + oom(np.max(np.var(y)))
        lst_oom = [10**ii for ii in range(max_oom)]
        params = pd.DataFrame(lst_oom, columns=['variance'])

        loglikelihoods = []
        
        for _,(i_var) in params.iterrows():
            kern = RBF(1, ARD=False, variance=i_var, lengthscale=lengthscale)
            m = GPRegression(x,y,kern)
            m.optimize()
            loglikelihoods.append(m.objective_function())
        
        params = params.join(pd.DataFrame(data=loglikelihoods,columns=['LogLikelihood']))
        params = params.sort_values('LogLikelihood')
        variance = params.iloc[0,:].variance

    return variance, lengthscale


def buildKernel(ndim,x,y,ARD=False):

    variance, lengthscale = initHyperParams(ndim, x, y)

    if (ndim > 1) and (not ARD):
        kern = buildCompositeKernel(ndim,variance,lengthscale)

    elif (ndim > 1):
        kern = RBF(ndim,ARD=True,variance=variance,lengthscale=lengthscale)

    elif (ndim == 1):
        kern = RBF(ndim,ARD=False,variance=variance,lengthscale=lengthscale)

    return kern


def buildCompositeKernel(ndim,variance=1,lengthscale=1):
    '''
    Build an RBF kernel that extends beyond the first dimension. 
        For example, if GP(X,Y) and X has three dimensions, first is Time which 
        is a continous variable and 2nd and 3rd are categorical, then, 
        K = K1 * (K2+K3) = K1*K2 + K1*K3 where K1 will have lengthscale specific for time. 
        
        See also https://github.com/ptonner/gp_growth_phenotype/bgreat.py 
    '''

    kern = RBF(1,active_dims=[0],ARD=False,lengthscale=lengthscale,variance=variance)
    ksum = None

    for ii in range(1,ndim):
        if ksum is None:
            ksum = RBF(1,active_dims=[ii],ARD=False)
        else:
            ksum += RBF(1,active_dims=[ii],ARD=False)

    if ksum is None:
        return kern
    else:
        return kern*ksum


def addFixedKernel(kern,y_dim,error):
    '''
    Empirically estimate time-variant noise and model as a fixed kernel.
        This will results in an optimized GP model with low Gaussian noise hyperparameter, 
        but yet incoporates empirical noise in the optimization of hyperparameters.

    Args:
        kern (GPy.kern.src.rbf.RBF)
        ndim (int): number of dimensions of output variables, typically OD is 1-dimnsional
        error (numpy.ndarray): known variance (i.e. error) for each observation output variable 

    Ret:
        kern (GPy.kern.src.add.Add)
    '''

    cov = np.eye(len(error))*error # ndim x ndim 
    fixed = Fixed(y_dim,cov)
    kern = kern + fixed

    return kern