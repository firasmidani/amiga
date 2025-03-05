#!/usr/bin/env python

'''
AMiGA library fo the microbial growth curve class.
'''

__author__ = "Firas S Midani"
__email__ = "midani@bcm.edu"

# TABLE OF CONTENTS (2 functions and 1 class with 13 sub-functions)

# linearize
# maxValueArg
#
# Curve (CLASS)
# 	__init__
#   compute_mse
#   log_to_linear
#   describe
#   AreaUnderCurve
#   CarryingCapacity
#   MaxGrowthRate
#   MinGrowthRate
#   StationaryDelta
#   LagTime
#   data
#   sample
#   plot

import numpy as np # type: ignore
import pandas as pd # type: ignore
import matplotlib.pyplot as plt # type: ignore

from scipy.stats import norm # type: ignore

from .diauxie import detectDiauxie
from .utils import getValue


def linearize(arr,baseline,floor=True,logged=True):
    '''
    Converts arr from log space to linear space.

    Args:
        arr (numpy.ndarray): typically log OD measurmenets over time
        baseline (float): should be baseline OD at first time point
        floor (float): if you want to shift curves to start at zero, pass True, else False

    Returns:
        self.predicted_OD (numpy.ndarray): same shape as input
    '''

    if arr is None:
        return None

    # add ln OD(0) and exponentiate
    if logged:
        arr = np.exp(arr+np.log(baseline))
    elif not logged:
        arr = arr + baseline

    # subtract OD(0)
    if floor:
        arr = arr - arr[0] 

    return arr


def maxValueArg(x,y):
    '''
    Find the maximum value (of y) and argument (of x) that maximizes it.

    Args:
    	x (np.ndarray)
    	y (np.ndarray)

    Returns:
        maxValue (float): x value corresponding to maximum value of y
        maxArg (float): maximum value of y
    '''

    assert x.shape[0] == y.shape[0]
    assert x.shape[1] == 1
    assert y.shape[1] == 1

    ind = np.where(y==y.max())[0][0]
    maxValue = y[ind,0]
    maxArg = float(x[ind][0])

    return maxValue, maxArg


class GrowthCurve:

    def __init__(self,x=None,y=None,y0=None,y1=None,y2=None,cov0=None,cov1=None,baseline=1.0,name=None,logged=True):
        '''
        Data structure for a growth curves. This is primarily used for computing 
            growth curve parameters and characteistics or converting curve and its fit to 
            real space. 

        Args:
            x (numpy.ndarray): independent variable N x D, where N is number of measurements, and D is number of dimesions
            y (nump.ndaray): dependent variable Nx1, typically log OD (input to GP model)
            y0 (numpy.ndarray): dependent variable N x 1, typically log OD (outupt of GP model)
            y1 (numpy.ndarray): first-order derivative of y0, typically d/dt log OD
            y2 (numpy.ndarray): second-order derivative of y1, typically d^2/dt^2 log OD
            cov0 (numpy.ndarray): covariance matrix for dependent variable y0
            cov1 (numpy.ndarray): covariance matrix for first-order derivatie y1
            baseline (float): the OD at time zero for dependent variable, y
        '''

        # verify data types
        for arg,value in [('x',x),('y0',y0),('y1',y1)]:
            assert isinstance(value,np.ndarray), f"{arg} must be a numpy ndaray"

        assert x.shape[1] == 1, "x must be one-dimensional"
        assert y0.shape == y1.shape, "y0 and y1 must have the same shape"

        if y2 is not None:
            assert isinstance(y2,np.ndarray), "y2 must be a numpy ndarray"
            assert y0.shape == y1.shape == y2.shape, "y2 must have the same shape as y0 and y1"

        # define attributes
        self.name = name
        self.baseline = baseline 
        self.logged = logged
        self.x = x
        self.y = y
        self.y0 = y0
        self.y1 = y1
        self.y2 = y2
        self.cov0 = cov0
        self.cov1 = cov1

        # derive linear transformations of select attributes
        self.log_to_linear()

        # compute all growth parameters or characteristics
        self.describe()

    def compute_mse(self,pooled=False):
        '''
        Computes Mean Squared Error
        '''

        if pooled:
            self_data = self.data()
            y = self_data.GP_Input.values
            y_hat = [ii for ii in self_data.GP_Output.values if ~np.isnan(ii)]
            y_hat = y_hat * int(len(y)/len(y_hat))
        else:
            y = self.linear_input_raised
            y_hat = self.linear_output_raised

        mse = (1./y.shape[0]) * sum((y-y_hat)**2)

        return mse


    def log_to_linear(self):
        '''
        Converts actual and predicted log OD from log space to linear space.
        '''

        self.linear_input = linearize(self.y,baseline=self.baseline,floor=True,logged=self.logged)
        self.linear_input_raised = linearize(self.y,baseline=self.baseline,floor=False,logged=self.logged)
        self.linear_output = linearize(self.y0,baseline=self.baseline,floor=True,logged=self.logged)
        self.linear_output_raised = linearize(self.y0,baseline=self.baseline,floor=False,logged=self.logged)


    def describe(self):

        dx_ratio_min = getValue('diauxie_ratio_min')
        dx_ratio_varb = getValue('diauxie_ratio_varb')

        self.AreaUnderCurve()
        self.CarryingCapacity()
        self.MaxGrowthRate()
        self.MinGrowthRate()
        self.LagTime()
        self.StationaryDelta()
        
        params = {'auc_lin':self.auc_lin,
                  'auc_log':self.auc_log,
                  'k_lin':self.K_lin,
                  'k_log':self.K_log,
                  't_k':self.t_K,
                  'gr':self.gr,
                  'dr':self.dr,
                  'td':self.td,
                  't_gr':self.t_gr,
                  't_dr':self.t_dr,
                  'death_lin':self.death_lin,
                  'death_log':self.death_log,
                  'lagC':self.lagC,
                  'lagP':self.lagP}

        if self.y2 is not None: 

            dx = detectDiauxie(self.x,self.y0,self.y1,self.y2,self.cov0,self.cov1,
                              thresh=dx_ratio_min,varb=dx_ratio_varb)

            # describe all phases
            df_dx = []
            for idx,row in dx.iterrows():
                t0,t1 = row['t_left'],row['t_right'] # indices
                t0,t1 = (np.where(self.x==ii)[0][0] for ii in [t0,t1]) # time at indices
                if (t0 == 0) and (t1==(len(self.x)-1)):
                    dx_params = params
                    dx_params['t0'] = row['t_left']
                    dx_params['tf'] = row['t_right']
                    df_dx.append(pd.DataFrame(dx_params,index=[idx]))
                else:
                    curve = GrowthCurve(x=self.x[t0:t1],
                                        y0=self.y0[t0:t1]-self.y0[t0],
                                        y1=self.y1[t0:t1],
                                        cov0=self.cov0[t0:t1,t0:t1],
                                        cov1=self.cov1[t0:t1,t0:t1])
                    dx_params = curve.params
                    dx_params['t0'] = row['t_left']
                    dx_params['tf'] = row['t_right']
                    df_dx.append(pd.DataFrame(dx_params,index=[idx]))

            df_dx = pd.concat(df_dx,axis=0)
            df_dx.columns = [f'dx_{ii}' for ii in df_dx.columns]

            params.update({'diauxie':[1 if dx.shape[0] > 1 else 0][0],'df_dx':df_dx})

        self.params = params


    def AreaUnderCurve(self):
        '''
        Computes the Area Under the Curve (AUC).
        '''

        dt = np.mean(self.x[1:,0]-self.x[:-1,0])  # time interval (int)

        mu_Log = self.y0 # get log or linear OD
        mu_Lin = self.linear_output # get log or linear OD

        D_Log = np.repeat(dt,mu_Log.shape[0]).T  # area under each interval (np.ndarray), size is (x.shape[0],)
        D_Lin = np.repeat(dt,mu_Lin.shape[0]).T  # area under each interval (np.ndarray), size is (x.shape[0],)        

        self.auc_lin = np.dot(D_Lin,mu_Lin)[0] # cumulative sum of areas
        self.auc_log = np.dot(D_Log,mu_Log)[0] # cumulative sum of areas


    def CarryingCapacity(self):
        '''
        Computes the maximum carrying capacity.
        '''    

        #print(self.x)
        self.K_log, self.t_K = maxValueArg(self.x,self.y0)
        self.K_lin, self.t_K = maxValueArg(self.x,self.linear_output)


    def MaxGrowthRate(self):
        '''
        Computes the maximum specific growth rate (gr) and generation doubling Time (td).
        '''

        self.gr, self.t_gr = maxValueArg(self.x,self.y1)

        # compu
        if self.gr == 0:
            self.td = np.inf
        else:
            self.td = (np.log(2.0))/self.gr


    def MinGrowthRate(self,after_max=True):
        '''
        Computes the minimum of the derivative, which will often be the maximum death rate (dr). 
        '''

        x_K = int(np.where(self.x[:,0]==self.t_K )[0]) # index (not time) at maximum growth

        mu = self.y1
        if after_max:
            mu = mu[x_K:,0]

        x_dr = np.where(mu==mu.min())[0][0]
        t_dr = float(self.x[x_dr+x_K][0])

        minGrowthRate = mu[x_dr]

        self.dr, self.t_dr = minGrowthRate, t_dr


    def StationaryDelta(self):
        '''
        Computes difference between carrying capcaity and final OD.
        '''
        
        if self.K_lin <= 0:
            self.death_log = 0
            self.death_lin = 0
        else: 
            self.death_log = np.abs(self.y0[-1][0] - self.K_log)
            self.death_lin = np.abs(self.linear_output[-1][0] - self.K_lin)


    def LagTime(self):
        '''
        Computes the lag time either the classical definition or a probabilistic definition.
            The former defines the lag time as the intersection with the axis parallel to time 
            of the tangent intersecting the derivative of the latent function at maximum growth. 
            This tangent has slope m equivalent to the maximum of the derivative of the latent.
            The latter defines lag time as the time at which the 95-percent credible interval of  
            the growth rate (i.e. derivative of latent) deviates from zero. 

        Args:
            mode (str): either 'Classical' or 'Probabilistic
            threshold (float): Confidence Interval, used for probabilistic inference of lag time.
        '''

        x = self.x
        y0 = self.y0
        y1 = self.y1
        cov1 = self.cov1

        # CLASSICAL MODE

        t_gr = self.t_gr # time at maximal growth rate
        x_gr = int(np.where(x[:,0]==t_gr)[0]) # index at maximal growth rate

        m1 = y1[x_gr] # slope at maximal growth rate
        m0 = y0[x_gr] # log OD at maximal growth rate

        if m1 == 0:
            lagC = np.inf # no growth, then infinite lag
        else:
            lagC = (t_gr - (m0/m1))[0]

        # PROBABILISTIC MODE

        confidence = getValue('confidence_adapt_time')

        prob = np.array([norm.cdf(0,m,np.sqrt(v)) for m,v in zip(y1[:,0],np.diag(cov1))])

        ind = 0
        while (ind < prob.shape[0]) and (prob[ind] > confidence):
            ind += 1

        if ind == prob.shape[0]:
            lagP = np.inf
        else:
            lagP = float(self.x[ind][0])

        self.lagC = lagC
        self.lagP = lagP


    def data(self):
        '''
        Summarizes the object's data, including estimates of best fit for data and its derivative using GPs.

        Args:
            sample_id (varies): can possibly be float, int, or str
        Returns:
            df (pandas.DataFrame): each row is a specific time-point. Identifying columns is Time (and possibly Sample_ID).
                Data columns include
                + OD_Fit (gp model fit of original data)
                + OD_Derivative (gp model fit of derivative, insensitive to y-value, i.e. whether OD is centered)
                + GP_Input (input to gp.GP() object), this is usually log-transformed and log-baseline-subtracted
                + GP_Output (output of gp.GP().predict()), hence also log-trasnformed and log-baseline-subtracted
                + OD_Growth_Data (GP_Input but converted to real OD and centered at zero)
                + OD_Growth_Fit (GP_Output but converted to real OD and centered at zero)      
        '''

        gp_time = self.x.ravel()
        gp_input = self.y.ravel()
        gp_output = self.y0.ravel()
        gp_derivative = self.y1.ravel()

        od_growth_data = self.linear_input.ravel()
        od_growth_fit = self.linear_output.ravel()
        od_fit = self.linear_output_raised.ravel()

        data = [gp_time,
                gp_input,
                gp_output,
                gp_derivative,
                od_growth_data,
                od_growth_fit,
                od_fit,
                ]

        labels = ['Time',
                  'GP_Input',
                  'GP_Output',
                  'GP_Derivative',
                  'OD_Growth_Data',
                  'OD_Growth_Fit',
                  'OD_Fit',]

        data = pd.DataFrame(data,index=labels).T

        if self.name is not None:
            sample_id = pd.DataFrame([self.name]*data.shape[0],columns=['Sample_ID'])
            data = sample_id.join(data)

        return data


    def sample(self):
        '''
        Sample the posterior distribution of the latent function and its derivative 
            n times, estimate growth parametes for each sample, then summarize with 
            mean and standard deviation. 
        '''

        n = getValue('n_posterior_samples')

        samples0 = np.random.multivariate_normal(self.y0.ravel(),self.cov0,n)
        samples1 = np.random.multivariate_normal(self.y1.ravel(),self.cov1,n)

        list_params = []

        for ii,y0,y1 in zip(range(n),samples0,samples1):

            y0_ii = y0[:,np.newaxis]
            y1_ii = y1[:,np.newaxis]

            curve_ii = GrowthCurve(x=self.x,y=self.y,y0=y0_ii,y1=y1_ii,
                                   cov0=self.cov0,cov1=self.cov1)
            list_params.append(curve_ii.params)

        df_params = pd.DataFrame(list_params)
        df_params_avg = df_params.mean(numeric_only=True)
        df_params_std = df_params.std(numeric_only=True)

        df_params_avg.index = [f'mean({ii})' for ii in df_params_avg.index]
        df_params_std.index = [f'std({ii})' for ii in df_params_std.index]

        self.posterior = pd.concat([df_params_avg,df_params_std]).to_dict()

        return self


    def plot(self,ax_arg=None):

        if not ax_arg:
            fig,ax = plt.subplots(2,1,figsize=[6,8],sharex=True)
        else:
            ax = ax_arg

        t = self.x.ravel()
        y =  self.y.ravel()
        y0 = self.y0.rave()
        y1 = self.y1.ravel()

        xmin = 0
        xmax = int(np.ceil(t[-1]))

        ax[0].plot(t,y,lw=5,color=(0,0,0,0.65))
        ax[0].plot(t,y0,lw=5,color=(1,0,0,0.65))
        ax[1].plot(t,y1,lw=5,color=(0,0,0,0.65))

        [ii.set(fontsize=20) for ii in ax[0].get_xticklabels()+ax[0].get_yticklabels()]
        [ii.set(fontsize=20) for ii in ax[1].get_xticklabels()+ax[1].get_yticklabels()]

        ylabel = getValue('hypo_plot_y_label')
        ax[1].set_xlabel('Time',fontsize=20)
        ax[0].set_ylabel(ylabel ,fontsize=20)
        ax[1].set_ylabel(f'd/dt {ylabel}',fontsize=20)

        ax[0].set_xlim([xmin,xmax])

        if not ax_arg:
            return fig,ax
        else:
            return ax

