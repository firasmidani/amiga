#!/usr/bin/env python

'''
DESCRIPTION library for Gaussian Process inference
'''

__author__ = "Firas Said Midani"
__version__ = "0.1.0"
__email__ = "midani@bcm.edu"

# TABLE OF CONTENTS


import GPy


def GP(x,y):
    '''
    

    Args:
        x
        y


    Returns:
        m 
    '''

    input_dim = x.shape[1]  # if time only, then 1 ?

    kern = GPy.kern.RBF(input_dim,ARD=True)  # Radial Basis Function

    m = GPy.models.GPRegression(x,y,kern)
    m.optimize

    return m


