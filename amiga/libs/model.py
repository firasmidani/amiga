#!/usr/bin/env python

'''
AMiGA library for the Gaussian Process Model class.
'''

__author__ = "Firas S Midani"
__email__ = "midani@bcm.edu"

# TABLE OF CONTENTS (1 function and 1 class with 7 sub-functions)

# describeVariance
# 
# Model (CLASS)
#   __init__
#   permute
#   fit
#   predict_y0
#   predict_y1
#   predict_y2
#   run

import warnings
import numpy as np # type: ignore
import pandas as pd # type: ignore

from GPy.models import GPRegression # type: ignore

from scipy.ndimage import filters # type: ignore

from .kernel import buildKernel,addFixedKernel
from .curve import GrowthCurve
from .utils import uniqueRandomString, subsetDf, getValue

if getValue('Ignore_RuntimeWarning'):
    warnings.filterwarnings("ignore", category=RuntimeWarning) 


def describeVariance(df,time='X0',od='Y'):
    '''
    df columns ['X0','X1',...,'Y']
    values of Xs except fo X0 should be non-unique
    '''

    window = getValue('variance_smoothing_window')
    nX = len(df[time].drop_duplicates())
    if window < 1:
        window = int(np.ceil(nX*window))
    
    df = df.sort_values(time)
    df.reset_index(drop=True,inplace=True)

    error = df[[time,'OD']]
    error = error.groupby([time]).apply(lambda x: np.nanvar(x.OD))
    error = pd.DataFrame(error,columns = ['error'])
    error = error.reset_index()
    error = error.drop_duplicates().set_index(time).sort_index()
    error.loc[:,'error'] = filters.gaussian_filter1d(error.error.values,window)
    df = pd.merge(df,error,on=time,how='outer').sort_values([time])
    
    return df


class GrowthModel:


    def __init__(self,df=None,model=None,x_new=None,baseline=1.0,ARD=False,heteroscedastic=False,nthin=1,logged=True):
        '''
        Data structure for Gaussian Process regression and related parameter inference.

        Attributes:
            x (numpy.ndarray): independent variables (N x D), where N is the number of observations, and 
                D is the number of dimensions (or variables).
            y (numpy.ndarray): dependent variables (N x 1), where N is the number of observations, and 
                the only column is the dependent or obesrved variable (often Optical Density or OD).
            key (dict or pandas.DataFrame): dictionary (k) or pandas.DataFrame (1 x k) that describes 
                k experimental variables about sample. Must include 'OD_Baseline' and 'Fold_Change' variables.

        Notes:
            for growth curve analysis, it is assumed that y was log-transformeed and baseline-corrected. 

        '''

        if model:
            self.model = model
            self.x_new = x_new
            self.ARD = ARD
            self.baseline = baseline
            self.logged = logged
            self.y = None
            self.df = None
            return None

        self.df = df.copy()

        # create a dummy non-unique variable/column
        foo = uniqueRandomString(avoid=df.keys())
        df[foo] = [1]*df.shape[0]
        varbs = df.drop(labels=['Time','OD'],axis=1).drop_duplicates()

        # for each unique non-time variable, estimate variance
        new_df = []
        for idx,row in varbs.iterrows():
            sub_df = subsetDf(df,row.to_dict())
            sub_df = describeVariance(sub_df,time='Time',od='OD')
            new_df.append(sub_df)

        new_df = pd.concat(new_df,axis=0)
        new_df = new_df.drop(labels=[foo],axis=1)

        # construct a thinner dataframe to speed up regression
        time = new_df.Time.sort_values().unique()
        time = time[::int(nthin)]
        thin_df = new_df[new_df.Time.isin(time)]
        
        # predictions of error and new are based on full dataframe
        tmp = new_df.drop(labels=['OD'],axis=1).drop_duplicates()
        error_new = tmp.loc[:,['error']].values
        x_new = tmp.drop(labels=['error'],axis=1).values

        # regression are based on input/output/error from thinned dataframe
        x = thin_df.drop(labels=['OD','error'],axis=1).values
        y = thin_df.loc[:,['OD']].values
        error = thin_df.loc[:,['error']].values
        x_keys = thin_df.drop(labels=['OD','error'],axis=1).keys()

        # save attributes
        self.x_keys = x_keys
        self.x_new = x_new
        self.x = x
        self.y = y 
        self.error = error
        self.error_new = error_new
        self.baseline = baseline
        self.logged = logged

        self.model = model
        self.ARD = ARD
        self.heteroscedastic = heteroscedastic
        self.noise = None


    def permute(self,varb=None):

        # get model input
        x = self.x.copy()
        y = self.y.copy()

        # shuffle targeet variable
        col = np.where(varb==self.x_keys)[0] # which one?
        shuffled = np.random.choice(x[:,col].ravel(),size=x.shape[0],replace=False)
        x[:,col] = shuffled[:,np.newaxis] # replace

        # same steps as fit, below can be done more concisely, but I worry about minor differenes
        x_dim = x.shape[1]
        y_dim = y.shape[1]

        kern = buildKernel(x_dim,x,y,ARD=self.ARD)
        mcopy = GPRegression(x,y,kern)
  
        if self.heteroscedastic:
            kern = addFixedKernel(kern,y_dim,self.error)
            mcopy = GPRegression(x,y,kern)
     
        mcopy.optimize()

        return mcopy.log_likelihood()


    def fit(self):

        if self.model:
            self.noise = self.model.Gaussian_noise.variance[0]
            return None

        x_dim = self.x.shape[1] # number of input dimensions, 1 if only time
        y_dim = self.y.shape[1] # number of ouptut dimensions, typically only 1 for log OD

        kern = buildKernel(x_dim,self.x,self.y,ARD=self.ARD)
        m = GPRegression(self.x,self.y,kern)

        if self.heteroscedastic:
            kern = addFixedKernel(kern,y_dim,self.error)
            m = GPRegression(self.x,self.y,kern)

        m.optimize()

        self.noise = m.Gaussian_noise.variance[0] # should be negligible (<1e-10) for full model

        if self.heteroscedastic:
            m.kern = m.kern.parts[0] # cannot predict with fixed kernel, so remove it

        self.model = m


    def predict_y0(self):

        x = self.x_new
        m = self.model

        mu,cov = m.predict(x,full_cov=True,include_likelihood=False)

        self.y0 = mu
        self.cov0 = cov

    def predict_y1(self):

        x = self.x_new
        m = self.model

        ndim = x.shape[1]

        if (ndim > 1) and (not self.ARD):
            mu,cov = (m.predictive_gradients(x)[0],None)
        else:
            mu,cov = m.predict_jacobian(x,full_cov=True)
            cov = cov[:,:,0,0]

        self.y1 = mu[:,0]
        self.cov1 = cov


    def predict_y2(self):

        x = self.x_new
        y = self.y1

        m = GPRegression(x,y)
        m.optimize()
        mu,cov = (m.predictive_gradients(x)[0],None)

        self.y2 = mu[:,0]
        self.cov2 = cov


    def run(self,simple=False,name=None,predict=True):

        self.fit()

        if not predict:
            return self, self.model.log_likelihood()

        self.predict_y0()
        self.predict_y1()
        self.predict_y2()

        if self.df is not None:
            actual_input = self.df.OD.values[:,np.newaxis]
        else:
            actual_input = None

        if not simple:
            time = self.x_new[:,0][:,np.newaxis]
            curve = GrowthCurve(x=time,y=actual_input,y0=self.y0,y1=self.y1,y2=self.y2,
                                cov0=self.cov0,cov1=self.cov1,
                                baseline=self.baseline,name=name,logged=self.logged)
            return curve
