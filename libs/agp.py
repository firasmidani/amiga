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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import norm
from scipy.signal import find_peaks,peak_prominences

sns.set_style('whitegrid')

class GP(object):

    def __init__(self,x=None,y=None):
        '''
        Data structure for Gaussian Process regression and related parameter inference.

        Attributes:
            x (pands.DataFrame): independent variables (N x D), where N is the number of observations, and 
                D is the number of dimensions (or variables)
            y (pandas.DataFrame): dependent variables (N x 1), where N is the number of observatiosn, and 
                the only column is the dependent or obesrved variable (often Optical Density or OD)
        '''

        self.x = x
        self.y = y

        self.dim = x.shape[1] # number of variables

        assert type(x) == pd.DataFrame, "x must be a pandas DataFrame"
        assert type(y) == pd.DataFrame, "y must be a pandas DataFrame"
        
    def fit(self,optimize=True):
        '''
        Designs a Gaussian Process mdoel for regression and optimizes it by maximizing log likelihood.

        Returns:
             m (GPY.model.gp_regression object)
        '''

        # define radial basis function kernel
        kern = GPy.kern.RBF(self.dim,ARD=True)

        # generate a GP model for regression and optimize by maximizing log-likelihood
        self.model = GPy.models.GPRegression(self.x.values,self.y,kern)

        if optimize:
            self.model.optimize()

        return self.model


    def derivative(self):
        '''
        Computes the derivatives of the predicted latent function with respect to object's X. 
        '''

        model = self.model
        x = self.x.values

        # from Sloak et a l. <-- from Tonner et al. (github.com/ptonner/gp_growth_phenotype/)
        mu,_ = model.predictive_gradients(x)
        _,cov = model.predict(x,full_cov=True)

        # covariance multiplier
        l = model.kern.lengthscale
        mult = [[((1./l)*(1-(1./l)*(y-z)**2))[0] for y in x] for z in x]
        cov = mult*cov

        self.derivative = (mu,cov)

        return self.derivative


    def predict(self):
        '''
        Infers GP estimate using optimized model. 
        ''' 

        model = self.model
        x = self.x.values

        # mu: np.ndarray (x.shape[0],1), cov: np.ndarray (x.shape[0],x.shape[0])
        mu,cov = model.predict(x,full_cov=True)

        self.predict = (mu,cov)

        return self.predict


    def auc(self):
        '''
        Infers the Area Under the Curve using GP estimate.
        '''

        x = self.x.values

        # mu: np.ndarray (x.shape[0],1), cov: np.ndarray (x.shape[0],x.shape[0])
        mu,cov = self.predict

        dt = np.mean(x[1:,0]-x[:-1,0])  # dt: int
        D = np.repeat(dt,mu.shape[0]).T  # D: np.ndarray (x.shape[0],)

        mu = np.dot(D,mu)[0]  # mu: int
        var = np.dot(D,np.dot(cov,D))  # var: int
       
        self.auc = (mu,var) 

        return self.auc


    def gr(self):
        '''
        Infers the maximmum specific growth rate using GP estimate.
        '''

        mu,cov = self.derivative
        ind = np.where(mu==mu.max())[0]

        mu = mu[ind,0,0][0]
        var = np.diag(cov)[ind][0]

        self.gr = (mu,var,ind)

        return self.gr


    def dr(self):
        '''Infers the maximum specific death rate using GP estimate.
        '''

        mu,cov = self.derivative
        ind = np.where(mu==mu.min())[0]

        mu = mu[ind,0,0][0]
        var = np.diag(cov)[ind][0]

        self.dr = (mu,var,ind)

        return self.dr 


    def k(self):
        '''
        Infers the carrying capacity using GP estimate.
        '''

        mu,cov = self.predict
        ind = np.where(mu==mu.max())[0]

        mu = mu[ind,0][0]
        var = np.diag(cov)[ind][0]

        self.k = (mu,var,ind)

        return self.k


    def td(self):
        '''
        Infers the doubling time using GP estimate of maximum specific growth rate.
        '''

        if not self.gr:
            msg = "USER ERROR: Object does not have a maximum specific growth rate (r) attribute. "
            msg += "Please call GP().gr() first before computing doubling time"
            print(msg)
            return None

        
        r = self.gr[0]
        td = (np.log(2.0)/r)*60.0

        self.td = td

        return self.td


    def lag(self,threshold=0.95):
        '''
        Infers the lag time using GP estimate.
        '''

        (mu,var) = self.derivative
        prob = np.array([norm.cdf(0,loc=m,scale=np.sqrt(v))[0] for m,v in zip(mu[:,:,0],var[:,0])])

        ind = 0
        while (ind < prob.shape[0]) and (prob[ind] > threshold):
            ind += 1
        if ind == prob.shape[0]:
            ind -= 1

        self.lag = float(self.x.values[ind])

        return self.lag


    def diauxie(self,ratio_max=0.25,x_as_time=True):
        '''
        Detects if diauxie occurs and determines the time at which growth phases are centered.

        Args:
            ratio_max (float): only peaks with ratio of counter height relative to maximum peak are called.
            x_as_time (boolean): return x-values either as time-points (True) or simply numerical index (False).
        '''

        # point to data needed for function
        time = self.x.values
        y_fit = np.ravel(self.predict[0])
        y_derivative = np.ravel(self.derivative[0])

        # find peak and vallies
        peaks,_ = callPeaks(y_derivative,0.2)
        valleys,_ = callPeaks(1-y_derivative,0.2)
        valleys = valleys + [0]  # useful below, if no valley left of peak, it will be instead zero


        print(peaks,valleys)
        # find heights
        x_ind,y_ind = [],[]
        for peak in peaks:

            # time difference to valleys
            lefts = [ii for ii in valleys if ii < peak]
            smallest_diff = np.max([ii-peak for ii in lefts])
            valley = peak + smallest_diff

            # find adjacent vallies
            #left = [ii if ii < peak else 0 for ii in valleys][0]
            #right = [ii if ii > peak else -1 for ii in valleys][0]

            # find y-values at x-values
            #left, center,right = [y_fit[ii] for ii in [left,peak,right]]

            # find heights
            #height = np.max([center-left,center-right])
            height = y_fit[peak]-y_fit[valley]
            print('\t{},{},{}'.format(peak,valley,height))
            # pass x-values and heights
            x_ind.append(peak)
            y_ind.append(height)

        print(x_ind,y_ind)
        # get height ratios
        y_ind = list(y_ind / np.max(y_ind))

        print(y_ind)
        # only keep peaks with height that are at least a certain ratio relative to maximum peak
        peaks = []
        for x,y in zip(x_ind,y_ind):
            if y > ratio_max:
                peaks.append(x)
        print(peaks)

        # save either as indices or time-points
        if x_as_time:
            self.peaks= list(np.ravel(time[peaks]))
        else:
            self.peaks = list(np.ravel(peaks))

        self.diauxie = [1 if len(self.peaks)>1 else 0][0]
        print() 
        return self.diauxie, self.peaks


    def describe(self,diauxie=0.25):

        params = {}

        self.fit()
        self.predict()
        self.derivative()

        params['auc'] = self.auc()[0]
        params['k'] = self.k()[0]
        params['gr'] = self.gr()[0]
        params['dr'] = self.dr()[0]
        params['td'] = self.td()
        params['lag'] = self.lag()
        params['diauxie'],params['peaks'] = self.diauxie(ratio_max=diauxie,x_as_time=True)

        self.params = params

        return self.params


    def data(self,sample_id=None):


        time = self.x
        time.columns = ['Time']

        od_data = self.y
        od_data.columns = ['OD']

        od_pred = pd.DataFrame(self.predict[0],columns=['Fit'])
        od_derivative = pd.DataFrame(np.ravel(self.derivative[0]),columns=['Derivative'])

        if sample_id is not None:
            sid = pd.DataFrame([sample_id]*time.shape[0],columns=['Sample_ID'])
            df = sid.join(time).join(od_data).join(od_pred).join(od_derivative)
        else:
            df = time.join(od_data).join(od_pred).join(od_derivaive)

        return df


    def plot(self,ax_user=None):

        # if user did not pass an axis
        if not ax_user:
            fig,ax = plt.subplots(2,1,figsize=[6,8],sharex=True)
        else:
            ax = ax_user


        time = self.x.values
        data = self.y.values
        pred = self.predict[0]
        derv = np.ravel(self.derivative[0])

        xmin = 0
        xmax = int(np.ceil(time[-1]))

        ax[0].plot(self.x.values,data,lw=5,color=(0,0,0,0.65))
        ax[0].plot(self.x.values,pred,lw=5,color=(1,0,0,0.65))
        ax[1].plot(self.x.values,derv,lw=5,color=(0,0,0,0.65))

        [ii.set(fontsize=20) for ii in ax[0].get_xticklabels()+ax[0].get_yticklabels()]
        [ii.set(fontsize=20) for ii in ax[1].get_xticklabels()+ax[1].get_yticklabels()]

        ax[1].set_xlabel('Time',fontsize=20)
        ax[0].set_ylabel('Optical Density',fontsize=20)
        ax[1].set_ylabel('d/dt Optical Density',fontsize=20)

        ax[0].set_xlim([xmin,xmax])
        ax[1].set_xlim([xmin,xmax])

        if not ax_user:
            return fig,ax
        else:
            return ax_user


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
    opt_model = GP(x,y).fit();  
    LL = opt_model.log_likelihood()[0]

    return LL


def callPeaks(x_data,ratio_max=0.2):
    '''
    Detects peaks in 1-dimensional data and calls them based on their height relative to primary peak

    Args:
        x_data (list or numpy.ndarray) of float values
        ratio_max (float): minimum allowed ratio of secondary peak heights to maximum primary peak height
    '''

    called_peaks = []
    called_heights = []

    # find all peaks
    peaks,_ = find_peaks(x_data)

    # find height of each peak where baseline is neigboring trough
    peak_heights = peak_prominences(x_data,peaks)[0]

    # find maximum peak, indicates primary growth phase
    max_height = np.max(peak_heights)

    # call rest of seconday peaks based on their ratio to maximum peak
    for pp,hh in zip(peaks,peak_heights):
        if hh > ratio_max*max_height:
            called_peaks.append(pp)
            called_heights.append(hh)

    return called_peaks,called_heights
