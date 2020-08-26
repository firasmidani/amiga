#!/usr/bin/env python

'''
AMiGA library for building kernels for Gaussian Process models.
'''

__author__ = "Firas S Midani"
__email__ = "midani@bcm.edu"


# TABLE OF CONTENTS

# buildKernel
# buildCompositeKernel
# addFixedKernel

import numpy as np

from GPy.kern import RBF,Fixed


def buildKernel(ndim,ARD=False):

    if (ndim > 1) and (ARD == False):
        kern = buildCompositeKernel(ndim)

    elif (ndim > 1):
        kern = RBF(ndim,ARD=True)

    elif (ndim == 1):
        kern = RBF(ndim,ARD=False)

    return kern


def buildCompositeKernel(ndim):
    '''
    Build an RBF kernel that extends beyond the first dimension. 
        For example, if GP(X,Y) and X has three dimensions, first is Time which 
        is a continous variable and 2nd and 3rd are categorical, then, 
        K = K1 * (K2+K3) = K1*K2 + K1*K3 where K1 will have lengthscale specific for time. 
        
        See also https://github.com/ptonner/gp_growth_phenotype/bgreat.py 
    '''

    kern = RBF(1,active_dims=[0],ARD=False)
    ksum = None

    for ii in range(1,ndim):
        if ksum is None: ksum = RBF(1,active_dims=[ii],ARD=False)
        else: ksum += RBF(1,active_dims=[ii],ARD=False)

    if ksum is None:  return kern
    else: return kern*ksum


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