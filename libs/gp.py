#!/usr/bin/env python

'''
AMiGA library for the Gaussian Process class for modelling growth curves and inferring growth paramters.
'''

__author__ = "Firas Said Midani"
__version__ = "0.2.0"
__email__ = "midani@bcm.edu"

# TABLE OF CONTENTS

# GP (CLASS)
#   __init__
#   fit
#   derivative
#   predict
#   predict_quantiles
#   estimate_auc
#   estimate_real_auc
#   estimeate_gr
#   estimate_dr
#   estimate_k
#   estimate_real_k
#   estimate_td
#   estimate_lag
#   estimate_diauxie
#   describe
#   data
#   plot
#
# convert_to_real_od
# buildRbfKernel
# callPeaks

import GPy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import norm
from scipy.signal import find_peaks,peak_prominences

sns.set_style('whitegrid')

class GP(object):

    def __init__(self,x=None,y=None,key=None):
        '''
        Data structure for Gaussian Process regression and related parameter inference.

        Attributes:
            x (pands.DataFrame): independent variables (N x D), where N is the number of observations, and 
                D is the number of dimensions (or variables).
            y (pandas.DataFrame): dependent variables (N x 1), where N is the number of observations, and 
                the only column is the dependent or obesrved variable (often Optical Density or OD).
            key (pandas.DataFrame): pandas.DataFrame (1 x k), that describes k experimental variables about sample.

        '''

        self.x = x
        self.y = y
        self.key = key

        self.dim = x.shape[1] # number of variables

        assert type(x) == pd.DataFrame, "x must be a pandas DataFrame"
        assert type(y) == pd.DataFrame, "y must be a pandas DataFrame"
        assert type(y) == pd.DataFrame, "y must be a pandas DataFrame"

        
    def fit(self,optimize=True):
        '''
        Designs a Gaussian Process mdoel for regression and optimizes it by maximizing log likelihood.

        Returns:
             m (GPY.model.gp_regression object)
        '''

        # define radial basis function kernel
        #kern = GPy.kern.RBF(self.dim,ARD=True)
        kern = buildRbfKernel(self.x)

        # generate a GP model for regression and optimize by maximizing log-likelihood
        self.model = GPy.models.GPRegression(self.x.values,self.y.values,kern)

        if optimize:
            self.model.optimize()

        return self.model


    def derivative(self):
        '''
        Computes the derivatives of the predicted latent function with respect to object's X.

        Returns:
            self.derivative_prediction () 
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

        self.derivative_prediction = (mu,cov)

        return self.derivative_prediction


    def predict(self,x=None):
        '''
        Infers GP estimate using optimized model. 

        Returns:
            self.prediction ()
        ''' 

        model = self.model

        if x is None:
            x = self.x.values

        # mu: np.ndarray (x.shape[0],1), cov: np.ndarray (x.shape[0],x.shape[0])
        mu,cov = model.predict(x,full_cov=True)

        self.prediction = (mu,cov)

        return self.prediction

    def predict_quantiles(self,x=None,quantiles=(2.5,50,97.5)):
        '''
        Get the predictive quantiles around the prediction at X.

        Args:
            x (numpy.ndarray) with size (X,1)

        Returns:
            self.quantiles (n-tuple) where each is a an numpy.ndarray with size (X,1)
                The predictions correspond to the quantiles arguments.
        ''' 

        model = self.model

        if x is None:
            x = self.x.values

        low,mid,high = model.predict_quantiles(x,quantiles=quantiles)

        self.quantiles = (low,mid,high)

        return self.quantiles


    def estimate_auc(self):
        '''
        Infers the Area Under the Curve using GP estimate.

        Returns:
            self.auc (float): area under the curve
        '''

        x = self.x.values

        # mu: np.ndarray (x.shape[0],1), cov: np.ndarray (x.shape[0],x.shape[0])
        mu,cov = self.prediction

        dt = np.mean(x[1:,0]-x[:-1,0])  # dt: int
        D = np.repeat(dt,mu.shape[0]).T  # D: np.ndarray (x.shape[0],)

        mu = np.dot(D,mu)[0]  # mu: int
        var = np.dot(D,np.dot(cov,D))  # var: int
       
        self.auc = (mu,var) 

        return self.auc


    def estimate_real_auc(self):
        '''
        Infers the Area Under the Curve using GP estimate.

        Returns:
            self.auc (float): area under the curve
        '''

        x = self.x.values

        # mu: np.ndarray (x.shape[0],1), cov: np.ndarray (x.shape[0],x.shape[0])
        mu,_ = self.prediction

        baseline = self.key.OD_Baseline.values[0]
        mu = convert_to_real_od(mu,baseline,floor=True)

        dt = np.mean(x[1:,0]-x[:-1,0])  # dt: int
        D = np.repeat(dt,len(mu)).T  # D: np.ndarray (x.shape[0],)

        auc = np.dot(D,mu) # mu: int
       
        return auc


    def estimate_gr(self):
        '''
        Infers the maximmum specific growth rate using GP estimate.

        Returns:
            self.gr (float): maximum specific growth rate
        '''

        mu,cov = self.derivative_prediction
        ind = np.where(mu==mu.max())[0]

        mu = mu[ind,0,0][0]
        var = np.diag(cov)[ind][0]

        self.gr = (mu,var,ind)

        return self.gr


    def estimate_dr(self):
        '''Infers the maximum specific death rate using GP estimate.

        Returns:
            self.dr (float): maximum specific death rate
        '''

        mu,cov = self.derivative_prediction
        ind = np.where(mu==mu.min())[0]

        mu = mu[ind,0,0][0]
        var = np.diag(cov)[ind][0]

        self.dr = (mu,var,ind)

        return self.dr 


    def estimate_k(self):
        '''
        Infers the carrying capacity using GP estimate.

        Returns:
            self.k (float): carrying capacity
        '''

        mu,cov = self.prediction

        ind = np.where(mu==mu.max())[0]

        mu = mu[ind,0][0]
        var = np.diag(cov)[ind][0]

        self.k = (mu,var,ind)

        return self.k

    def estimate_real_k(self):
        '''
        Infers the carrying capacity using GP estimate.

        Returns:
            self.k (float): carrying capacity
        '''

        mu,_ = self.prediction

        baseline = self.key.OD_Baseline.values[0]
        mu = convert_to_real_od(mu,baseline,floor=True)

        ind = np.where(mu==mu.max())[0]

        k = mu[ind][0]

        return k


    def estimate_td(self):
        '''
        Infers the doubling time using GP estimate of maximum specific growth rate.

        Returns:
            self.td (float): doubling time (time units for outupt is defined in config.py)
        '''

        if not self.gr:
            msg = "USER ERROR: Object does not have a maximum specific growth rate (r) attribute. "
            msg += "Please call GP().gr() first before computing doubling time"
            print(msg)
            return None

        
        r = self.gr[0]
        td = (np.log(2.0)/r)

        self.td = td

        return self.td


    def estimate_lag(self,threshold=0.95):
        '''
        Infers the lag time using GP estimate.

        Args:
            threshold (float): acceptable threshold for running variance of data.

        Returns:
            self.lag (float): lag time
        '''

        (mu,var) = self.derivative_prediction
        prob = np.array([norm.cdf(0,loc=m,scale=np.sqrt(v))[0] for m,v in zip(mu[:,:,0],var[:,0])])

        ind = 0
        while (ind < prob.shape[0]) and (prob[ind] > threshold):
            ind += 1
        if ind == prob.shape[0]:
            ind -= 1

        self.lag = float(self.x.values[ind])

        return self.lag


    def estimate_diauxie(self,ratio_min=0.25,fc_min=1.5,x_as_time=True):
        '''
        Detects if diauxie occurs and determines the time at which growth phases are centered.

        Args:
            ratio_min (float): only peaks with ratio of counter height relative to maximum peak are called.
            x_as_time (boolean): return x-values either as time-points (True) or simply numerical index (False).

        Returns:
            self.diauxie (int): binary {0,1}, whether diauxic shift detected for object or not.
            self.peaks (list of floats): time points at which peaks where detected.
        '''

        # get fold-change from key, used to call diauxie
        fold_change = self.key.Fold_Change.values[0]

        # point to data needed for function
        time = self.x.values
        y_fit = np.ravel(self.prediction[0])
        y_derivative = np.ravel(self.derivative_prediction[0])

        # find peak and vallies
        peaks,_ = callPeaks(y_derivative,0.2)

        valleys,_ = callPeaks(1-y_derivative,0.2)

        valleys = valleys + [0]  # useful below, if no valley left of peak, it will be instead zero

        # find heights
        x_ind,y_ind = [],[]
        for peak in peaks:

            # time difference to valleys
            lefts = [ii for ii in valleys if ii < peak]
            smallest_diff = np.max([ii-peak for ii in lefts])
            valley = peak + smallest_diff

            # find heights
            height = y_fit[peak]-y_fit[valley]

            # pass x-values and heights
            x_ind.append(peak)
            y_ind.append(height)

        # get height ratios
        y_ind = list(y_ind / np.max(y_ind))

        # only keep peaks with height that are at least a certain ratio relative to maximum peak
        peaks = []
        for x,y in zip(x_ind,y_ind):
            if y > ratio_min:
                peaks.append(x)

        # save either as indices or time-points
        if x_as_time:
            self.peaks= list(np.ravel(time[peaks]))
        else:
            self.peaks = list(np.ravel(peaks))

        self.diauxie = [1 if (len(self.peaks)>1) and (fold_change>fc_min)  else 0][0]

        return self.diauxie, self.peaks


    def describe(self,dx_ratio_min=0.25,dx_fc_min=1.5):
        '''
        Complete growth curve analysis that includes fitting data to a Gaussian Process, predicting best fit of data, 
        predicting best fit for derivative of data, and inferring growth curve kinetic parametrs.

        Args:
            diauxie (float): threshold used for calling diauxic shifts, see class function diauxie

        Returns:
            self.params (dictionary): keys are identifiers of growth curve parameters and values are their estimates.
                all are floats/ints except for 'peaks' which is a list of floats.
        '''

        params = {}

        self.fit()
        self.predict()
        self.derivative()

        params['auc'] = self.estimate_real_auc() #self.estimate_auc()[0]
        params['k'] = self.estimate_real_k() #self.estimate_k()[0]
        params['gr'] = self.estimate_gr()[0]
        params['dr'] = self.estimate_dr()[0]
        params['td'] = self.estimate_td()
        params['lag'] = self.estimate_lag()

        dx,dx_peaks = self.estimate_diauxie(ratio_min=dx_ratio_min,fc_min=dx_fc_min,x_as_time=True)
        params['diauxie'],params['peaks'] = dx,dx_peaks

        self.params = params

        return self.params


    def data(self,sample_id=None):
        '''
        Summarizes the object's data, including estimates of best fit for data and its derivative using GPs.

        Args:
            sample_id (varies): can possibly be float, int, or str
        Returns:
            df (pandas.DataFrame): rows are time points (t) and columns are 'Time','OD','Fit','Derivative',
                and possibly 'Sample_ID' (4-5).
        '''

        baseline = self.key.OD_Baseline.values[0]

        time = self.x
        time.columns = ['Time']

        od_data = convert_to_real_od(self.y,baseline=baseline,floor=True); 
        od_data.columns = ['OD']
        
        od_pred = pd.DataFrame(convert_to_real_od(self.prediction[0],baseline=baseline,floor=True),columns=['Fit'])
        od_derivative = pd.DataFrame(np.ravel(self.derivative_prediction[0]),columns=['Derivative'])

        if sample_id is not None:
            sid = pd.DataFrame([sample_id]*time.shape[0],columns=['Sample_ID'])
            df = sid.join(time).join(od_data).join(od_pred).join(od_derivative)
        else:
            df = time.join(od_data).join(od_pred).join(od_derivaive)

        return df


    def plot(self,ax_user=None):
        '''
        Plots two-panels that describe object's data, best fit based on GP estimate of data and its derivative.

        Returns:
            if ax_user was passed, returns ax_user, otherwise, return figure object and axis object.
        '''

        # if user did not pass an axis
        if not ax_user:
            fig,ax = plt.subplots(2,1,figsize=[6,8],sharex=True)
        else:
            ax = ax_user


        time = self.x.values
        data = self.y.values
        pred = self.prediction[0]
        derv = np.ravel(self.derivative_prediction[0])

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

        plt.close()


def convert_to_real_od(data,baseline,floor=True):
    '''
    Converts data in transformed space to original space (actual OD, not arbitrary units).

    Args:
        baseline (float): should be baseline OD at first time point
        floor (float): if you want to shift curves to start at zeero, pass an argument equivalent to OD(0)

    Returns:
        self.predicted_OD ()
    '''

    if isinstance(data,pd.DataFrame):

        y_data = np.ravel(data.values)

        y_real = [np.exp(yd+np.log(baseline)) for yd in y_data]

        y_real = pd.DataFrame(y_real,columns=data.columns)

    elif isinstance(data,np.ndarray) or isinstance(data,list):

        y_data = np.ravel(data)

        y_real = np.ravel([np.exp(yd+np.log(baseline)) for yd in y_data])


    if floor and isinstance(data,pd.DataFrame):

        y_real = y_real - y_real.iloc[0,:]

    elif floor and (isinstance(data,np.ndarray) or isinstance(data,list)):

        y_real = y_real - y_real[0]

    return y_real


def buildRbfKernel(x):

    size = x.shape[1]
    names = x.columns

    ret = GPy.kern.RBF(1,active_dims=[0],name=names[0],ARD=True)
    ksum = None

    for ii in range(1,size):
        if ksum is None:
            ksum = GPy.kern.RBF(1,active_dims=[ii],name=names[ii],ARD=True)
        else:
            ksum += GPy.kern.RBF(1,active_dims=[ii],name=names[ii],ARD=True)

    if ksum is None:
        return ret

    return ret*ksum


def callPeaks(x_data,ratio_min=0.2):
    '''
    Detects peaks in 1-dimensional data and calls them based on their height relative to primary peak

    Args:
        x_data (list or numpy.ndarray) of float values
        ratio_min (float): minimum allowed ratio of secondary peak heights to maximum primary peak height
    '''

    called_peaks = []
    called_heights = []

    # find all peaks
    peaks,_ = find_peaks(x_data)

    # if none detected, find maximum and return it
    if len(peaks)==0:
        peaks = list(np.where(x_data==np.max(x_data))[0])
        return peaks, None

    # find height of each peak where baseline is neigboring trough
    peak_heights = peak_prominences(x_data,peaks)[0]

    # find maximum peak, indicates primary growth phase
    max_height = np.max(peak_heights)

    # call rest of seconday peaks based on their ratio to maximum peak
    for pp,hh in zip(peaks,peak_heights):
        if hh > ratio_min*max_height:
            called_peaks.append(pp)
            called_heights.append(hh)

    return called_peaks,called_heights
