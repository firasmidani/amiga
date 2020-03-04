#!/usr/bin/env python

'''
DESCRIPTION library for Gaussian Process inference
'''

__author__ = "Firas Said Midani"
__version__ = "0.1.0"
__email__ = "midani@bcm.edu"

# TABLE OF CONTENTS

# GP
# computeLikelihood

import GPy
import pandas as pd

def GP(x,y):
    '''
    Designs a Gaussian Process mdoel for regression and optimizes it by maximizing log likelihood.

    Args:
        x (pands.DataFrame): independent variables (N x D), where N is the number of observations, and 
            D is the number of dimensions (or variables)
        y (pandas.DataFrame): dependent variables (N x 1), where N is the number of observatiosn, and 
            the only column is the dependent or obesrved variable (often Optical Density or OD)

    Returns:
        m (GPY.model.gp_regression object)
    '''

    # define number of dimensions
    input_dim = x.shape[1]

    # define radial basis function kernel
    kern = GPy.kern.RBF(input_dim,ARD=True)

    # generate a GP model for regression and optimize by maximizing log-likelihood
    m = GPy.models.GPRegression(x.values,y,kern)
    m.optimize()

    return m


def computeLikelihood(df,variables):
    '''
    Computes log-likelihood of a Gaussian Process Regression inference. 

    Args:
        df (pandas.DataFrame): N x p, where N is the number of individual observations (i.e.
            specific time measurement in specific well in specific plate), p must be include parameters
            which will be used as independent variables in Gaussian Process Regression. These variables 
            can be either numerical or categorical. Later will be converted to enumerated type. Variables
            must also include both OD and Time column with values of float type.
        variables (list of str): must be column headers in df argument.

    Returns:
        LL (float): log-likelihood
    '''

    # reduce dimensionality
    df = df.loc[:,['OD']+variables]
    df = df.sort_values('Time').reset_index(drop=True)  # I don't think that sorting matters, but why not

    # all variables must be encoded as an enumerated type (i.e. int or float)
    for variable in variables:
        if (variable == 'Time'):
            continue
        else:
            df.loc[:,variable] = pd.factorize(df.loc[:,variable])[0]

    # define design matrix
    y = pd.DataFrame(df.OD)
    x = pd.DataFrame(df.drop('OD',axis=1))

    # build and optimize model, then return maximized log-likelihood
    opt_model = GP(x,y);  
    LL = opt_model.log_likelihood()[0]

    return LL


