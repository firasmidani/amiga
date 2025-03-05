#!/usr/bin/env python

'''
AMiGA library of functions for object-oriented testing of hypotheses with GP regression.
'''

__author__ = "Firas S Midani"
__email__ = "midani@bcm.edu"

import os
import sys
import operator
import numpy as np # type: ignore
import pandas as pd # type: ignore
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore

from scipy.stats import norm, percentileofscore # type: ignore
from matplotlib import rcParams # type: ignore
from matplotlib.ticker import MultipleLocator # type: ignore
from tabulate import tabulate # type: ignore

from .detail import updateMappingControls, shouldYouSubtractControl
from .model import GrowthModel
from .growth import GrowthPlate
from .org import assembleFullName, assemblePath
from .trim import annotateMappings,trimMergeMapping, trimMergeData
from .utils import subsetDf, reverseDict
from .utils import getValue, getTimeUnits, getHypoPlotParams, selectFileName
from .comm import prettyNumberDisplay, smartPrint, tidyDictPrint, tidyMessage
from .params import initDiauxieList, initParamList, initParamDf, mergeDiauxieDfs
from .params import minimizeParameterReport, minimizeDiauxieReport
from .params import removeFromParameterReport, removeFromDiauxieReport
from .params import articulateParameters, prettyifyParameterReport
from .plot import addRealPlotLine, addMVNPlotLine, setAxesLabels
from .plot import dynamicWindowAdjustment, plotDeltaOD, savePlotWithLegends

pd.set_option('future.no_silent_downcasting', True)


class HypothesisTest:

    def __init__(self,mapping_dict=None,data_dict=None,params_dict=None,args_dict=None,directory_dict=None,sys_exit=True):
        '''
        Framework for hypothesis testing. 

        Args:
            mapping_dict (dictionary of mapping as pandas.DataFrames)
            data_dict (dictionary of data as pandas.DataFrames)
            params_dict (dictionary)
            args_dict (dictionary)
            directory_dict (dictionary)
            sys_exit (boolean)
        '''

        # define attibutes
        self.args = args_dict
        self.params = params_dict # need keys 'hypothesis' and 'subset' 
        self.directory = directory_dict
        self.hypothesis = params_dict['hypothesis']
        self.subtract_control = self.args.subtract_control
        self.subtract_blanks = self.args.subtract_blanks
        self.verbose = self.args.verbose

        # only proceed if hypothesis is valid
        if self.checkHypothesis():
            return None

        # undertake hypothesis test
        self.initPaths()
        self.defineMapping(mapping_dict=mapping_dict)
        self.addInteractionTerm()
        self.checkControlSamples(mapping_dict=mapping_dict)
        self.defineData(data_dict=data_dict)
        self.prettyTabulateSamples()
        self.prepRegressionPlate()
        self.tidifyRegressionData()
        self.factorizeCategoicals()
        self.executeRegression()
        self.generatePredictions()
        self.savePredictions()
        self.computeFullDifference()
        self.plotPredictions()
        self.reportRegression()
        self.exportReport()

        # bid user farewell
        if sys_exit:
            sys.exit(tidyMessage('AMiGA completed your request!'))


    def initPaths(self):
        '''
        Initialize paths for for saving data and results. 
        '''

        # if user did not pass file name for output, use time stamp
        file_name = selectFileName(self.args.output)
        dir_path = assemblePath(self.directory['models'],file_name,'')
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)      

        # running model on transformed results and recording results
        file_path_key = assembleFullName(dir_path,'',file_name,'key','.txt')
        file_path_input = assembleFullName(dir_path,'',file_name,'input','.txt')

        paths_dict={}

        paths_dict['filename'] = file_name
        paths_dict['dir'] = dir_path
        paths_dict['key'] = file_path_key
        paths_dict['input'] = file_path_input

        self.paths_dict = paths_dict


    def checkHypothesis(self):
        '''
        Verifies that a user provided a hypothesis ant that is meets the following crieteria. The alternative
            hypothesis must have only two variables, one being time.

        Args:
            hypothesis (dictionary): e.g. {'H0':['Time'],'H1':['Time','Substrate']}

        Returns
            hypotheis (dict): input argument but with 'Time' added to values, if initially missing
            variables (list): variables that differ between the null and alternative hypotheses
        '''

        hypothesis = self.hypothesis.copy()

        if len(hypothesis)==0:
            msg = 'USER ERROR: No hypothesis has been passed to AMiGA via either command-line or text file.\n'
            return True #exit

        # what is the variable of interest based on hypothesis?
        target = list(set(hypothesis['H1'])^(set(hypothesis['H0'])))

        if len(target) > 1:
            msg = 'FATAL USER ERROR: Your null and alternaive hypotheses differ by more than one variables. '
            msg += 'AMiGA can not test for differential growth between conditions that vary by more than one variable.\n'
            sys.exit(msg)
            #msg += 'All of these distinct variables will be permuted for credible interval testing.\n'
            #smartPrint(msg,True)

        # make sure that Time is included in both hypotheses
        if ('Time' not in hypothesis['H0']):
            hypothesis['H0'] = hypothesis['H0'] + ['Time']
            msg = 'USER WARNING: Time was missing from the null hypothesis (H0) but it will be included as a variable.\n'
            smartPrint(msg,True)

        if ('Time' not in hypothesis['H1']):
            hypothesis['H1'] = hypothesis['H1'] + ['Time']
            msg = 'USER WARNING: Time was missing from the alternative hypothesis (H1) but it will be included as a variable.\n'
            smartPrint(msg,True)

        # get all unique variables in hypothesis into a list
        variables = {i for value in hypothesis.values() for i in value}
        variables = variables.difference({'Time'})  # remove Time
        self.non_time_varbs = variables
        variables = ['Time'] + list(variables)  # add back Time to beginning of list, not necessary but helpful
        
        self.variables = variables
        self.target = target
        #self.hypothesis = hypothesis

        return False


    def defineMapping(self,mapping_dict=None):
        '''
        Trim data based on user-passed or default 'subest' and 'flag' parameters
        '''

        # annotate Subset and Flag columns in mapping files, then trim and merge into single dataframe
        mapping_dict = annotateMappings(mapping_dict,self.params,self.verbose)

        self.master_mapping = trimMergeMapping(mapping_dict,self.verbose) # named index: Sample_ID


    def addInteractionTerm(self):
        '''
        If user passed hypothesis with an interaction term (identified by an astersisk), then
            create 
        '''

        # add interaction term, if needed
        mapping = self.master_mapping

        for variable in self.target:

            if ('*' in variable):
                pairs = variable.split("*")
                var_dict = {}
                if ('(' in variable) and (')' in variable):

                    for pair in pairs:
                        var,ctrl = pair.split('(')
                        var_dict[var]=ctrl[:-1]

                    intx = subsetDf(mapping,var_dict).index.values
                    mapping.loc[:,variable] = [0]*mapping.shape[0]
                    mapping.loc[intx,variable] = [1]*len(intx)

                else:

                    df = mapping.loc[:,pairs].drop_duplicates()
                    df.loc[:,variable] = df.apply(lambda x: f'{x[pairs[0]]} x {x[pairs[1]]}',axis=1) 

                    mapping = pd.merge(mapping.reset_index(),df,on=pairs,how='left')
                    mapping = mapping.set_index('Sample_ID')

        self.master_mapping = mapping


    def checkControlSamples(self,mapping_dict=None):

        mm = self.master_mapping
        sc = self.subtract_control 

        if sc:
            sc = shouldYouSubtractControl(mm,self.target)
        
        mm = updateMappingControls(mm,mapping_dict,to_do=sc).dropna(axis=1)
        
        self.master_mapping = mm
        self.subtract_control = sc


    def defineData(self,data_dict=None):

        # grab all data
        self.master_data = trimMergeData(data_dict,self.master_mapping,self.args.skip_first_n,self.verbose) # unnamed index: row number


    def prettyTabulateSamples(self):

        if self.verbose:
            print_map = self.master_mapping.copy().drop(['Subset'],axis=1)
            tab = tabulate(print_map,headers='keys',tablefmt='psql')
            msg = 'The following samples will be used in hypothesis testing:'
            msg = f'\n{msg}\n{tab}\n'
            smartPrint(msg,self.verbose)
            smartPrint(tidyDictPrint({'Hypothesis is':self.hypothesis}),self.verbose)


    def prepRegressionPlate(self):
        '''
        Packages data into a growth.GrowthPlate() object and performs a select number of class functions.

        Args:
            data (pandas.DataFrame): t (number of measurements) by n+1 (number of samples + one column for time)
            mapping (pandas.DataFrame): n (number of samples) by p (number of variables)
            subtract_control (boolean)
            thinning_step (int): how many time points to skip between selected time points. 
        '''

        plate = GrowthPlate(self.master_data,self.master_mapping)
        plate.convertTimeUnits(input=getTimeUnits('input'),output=getTimeUnits('output'))
        plate.subtractControl(to_do=self.subtract_blanks,drop=True,blank=True)
        plate.subtractControl(to_do=self.subtract_control,drop=True,blank=False)
        plate.raiseData()  # replace non-positive values, necessary prior to log-transformation
        plate.logData(to_do=self.args.log_transform)
        plate.subtractBaseline(to_do=True,poly=getValue('PolyFit'),groupby=list(self.non_time_varbs))
        plate.dropFlaggedWells(to_do=True)
        plate.key.to_csv(self.paths_dict['key'],sep='\t',header=True,index=True)  # save model results
        self.plate = plate
        self.ntimepoints = plate.time.shape[0]


    def tidifyRegressionData(self):
        '''
        Prepares a single dataframe for running GP regression.

        Args:
            plate (growth.GrowthPlate)
            hypothesis (dictionary): has two keys 'H0' and 'H1', values are lists of variables
                which are headers of plate.ke pandas.DataFrame
            mthin (int)
            save_path (str): default is None

        Returns:
            data (pandas.DataFrame): long-format
            factor_dict (dictionary): maps unique values for variables to numerical integers
        '''

        save_path = self.paths_dict['input']

        variables = self.variables
        non_times = self.non_time_varbs

        # melt data frame so that each row is a single time measurement
        #   columns include at least 'Sample_ID' (i.e. specific well in a specific plate) and
        #   'Time' and 'OD'. Additioncal column can be explicilty called by user using hypothesis.

        data = (self.plate.time).join(self.plate.data)
        data = pd.melt(data,id_vars='Time',var_name='Sample_ID',value_name='OD')
        data = data.merge(self.plate.key,on='Sample_ID')

        # Handle missing data by dropping them (ie, replicates missing select time points)
        missing = np.unique(np.where(data.iloc[:,1:].isna())[0])
        missing = data.iloc[missing,].Time.values

        tmp = data.sort_values(['Time']).loc[:,['Time','OD']]
        drop_idx = tmp[tmp.isna().any(axis=1)].index
        data = data.drop(labels=drop_idx,axis=0)
        data = data.drop(labels=['Subset'],axis=1)

        if save_path:
            data.to_csv(save_path,sep='\t',header=True,index=True)

        # reduce dimensionality
        data = data.reindex(['OD']+variables,axis=1)
        data = data.sort_values('Time').reset_index(drop=True) # I don't think that sorting matters, but why not
        self.data = data

        value_counts = data.loc[:,list(non_times)].apply(lambda x: len(np.unique(x)),axis=0)
        if any(value_counts >2):
            value_counts = pd.DataFrame(value_counts).reset_index()
            value_counts.columns = ['Variable','# Unique Values']
            tab = tabulate(value_counts,headers='keys',tablefmt='psql')

            msg = '\n USER WARNING: Hypothesis testing should only be performed on binary conditions.'
            msg = f'{msg} See below.\n\n{tab}\n'
            sys.exit(msg)


    def factorizeCategoicals(self):

        data = self.data

        # factorize variable values and store mapping
        self.factor_dict = {}
        for varb in self.variables:
            if varb == 'Time':
                continue
            values = data[varb].unique()
            values_code = range(len(values))
            values_maps = {v:c for v,c in zip(values,values_code)}
            data.loc[:,varb] = data.loc[:,varb].replace(values_maps).infer_objects(copy=False)
            self.factor_dict[varb] = values_maps


    def executeRegression(self):
        '''
        Computes the log Bayes Factor and its null distribution (based on permutation tests).

        Args:
            data (pandas.DataFrame): each row is a single measurement (i.e. time point in a well), columns are variables
                and must include 'Time', 'OD'.
            hypothesis (dictionary): keys must be 'H0' and 'H1', values are lists of variables (must match data keys)
            nperm (int): number ofxec permutations to generate null distribution

        Returns:
            log_BF (float): log Bayes Factor = log (P(H1|D)/P(H0|D))
            null_distribution (list of floats): the null distribution for log Bayes Factor where variable of interest
                was permuted for a certain number of times (based on nperm).
        '''

        verbose = self.verbose
        hypothesis = self.hypothesis
        fix_noise = self.args.fix_noise
        nperm = self.args.number_permutations
        nthin = self.args.time_step_size

        data = self.data

        data0 = data.loc[:,['OD']+hypothesis['H0']]
        data1 = data.loc[:,['OD']+hypothesis['H1']]

        gm0 = GrowthModel(df=data0,ARD=True,heteroscedastic=fix_noise,nthin=nthin,logged=self.plate.mods.logged)
        gm1 = GrowthModel(df=data1,ARD=True,heteroscedastic=fix_noise,nthin=nthin,logged=self.plate.mods.logged)

        gm0,LL0 = gm0.run(predict=False)
        gm1,LL1 = gm1.run(predict=False)
        log_BF = LL1-LL0

        self.log_BF = log_BF
        self.model = gm1
        self.LL0 = LL0
        self.LL1 = LL1
        self.log_BF_null_dist = None

        null_distribution = []
        to_permute = list(set(hypothesis['H1']).difference(set(hypothesis['H0'])))[0]
        for rep in range(nperm):
            smartPrint(f'Permutation #{rep}',verbose)
            null_distribution.append(gm1.permute(to_permute)-LL0)
        smartPrint('',verbose)
        if null_distribution:
            self.log_BF_null_dist = null_distribution


    def generatePredictions(self):
        '''
        A Gaussian Process is modelled for Nx1 response (Y) variable and NxD explantory
            (X) variables, with multiple replicates for each unique set of D variables. 
            generateTestMatrix() identifies all uniques instances of X, disregarding the 
            time variable. Time is however reintroduced but as an evenly spaced vector
            based on pred_num_time_points variable in config.py.

        Args:
            model (GPy.models.gp_regression.GPRegression)
            model_input (list of str): variables must be keys for data
            data (pandas.DataFrame): samples by variables (includes Time)

        Returns:
            x (pandas.DataFrame): full design matrix (includes time)
            x_min (pandas.DataFrame): minimal design matrix (excludes time)
        '''

        data = self.data
        model = self.model
        model_input = self.hypothesis['H1']
        fix_noise = self.args.fix_noise

        # first, generates a dataframe where each row is a unique permutations of non-time  variables
        x = data.loc[:,model_input].drop(labels=['Time'],axis=1)
        x = x.drop_duplicates().reset_index(drop=True)
        x_min = x.copy()
        x['merge_key'] = [1]*x.shape[0]

        # second, sample evenly spaced time points based on min and max time points
        x_time = pd.DataFrame(data.Time.unique(),columns=['Time'])
        x_time['merge_key'] = [1]*x_time.shape[0]

        # combine time and variable permutation dataframes, such that each unique time point 
        #    is mapped to all unique permutations of variabels
        x = pd.merge(x_time,x,on='merge_key').drop(labels=['merge_key'],axis=1)

        if fix_noise:
            sigma_noise =np.ravel(model.error_new)+model.noise
        else:
            sigma_noise = np.ravel([model.noise]*x.shape[0])

        x_new = [tuple(ii) if len(ii)>1 else [ii] for ii in model.x_new]
        x = x.set_index(list(x.keys())).loc[x_new].reset_index() # reorder x to match noise

        # given gp model, predict mu and Sigma on experimental design input
        mu,cov = (model.model).predict(x.values,full_cov=True,include_likelihood=False)
        mu_var = pd.DataFrame([np.ravel(mu),np.ravel(np.diag(cov)),np.ravel(sigma_noise)],
                              index=['mu','Sigma','Noise']).T
        x = x.reset_index(drop=True).join(mu_var)

        self.x_full = x
        self.x_min = x_min


    def savePredictions(self):
        '''
        Given model predictions of growth curves (for each unique set of conditions tested),
            describe the latent function and its derivative in terms of growth parameters. 
            Reports results in a file with {file_name}_params name in dir_path directory. 

        Args:
            model (GPy.models.gp_regression.GPRegression)
            data (pandas.DataFrame)
            hypothesis (dictionary): e.g. {'H0':['Time'],'H1':['Time','Substrate']}
            actor_dict (dictionary): mapping of unique values of variables to numerical integers
            posterior (boolean)
            save_latent (boolean)
            dir_path (str): path to directory
            file_name (str): file name

        Returns:
            x_full (pandas.DataFrame): 
            x_min (pandas.DataFrame):

        '''

        model = self.model
        factor_dict = self.factor_dict
        variable = self.target[0]
        confidence = self.args.confidence  # confidence interval, e.g. 0.95

        posterior = self.args.sample_posterior
        save_latent = self.args.save_gp_data
        do_not_log_transform = self.args.do_not_log_transform

        dir_path = self.paths_dict['dir']
        file_name = self.paths_dict['filename']

        # define hypothesis paraameters
        x_full = self.x_full
        x_min = self.x_min

        diauxie_dict = {}
        params_latent = initParamDf(x_min.index,complexity=0)
        params_sample = initParamDf(x_min.index,complexity=1)

        for idx,row in x_min.iterrows():

            # get x and y data
            df = subsetDf(x_full.drop(labels=['mu','Sigma','Noise'],axis=1),row.to_dict())

            # get curve based on model predictions
            gm = GrowthModel(model=model.model,x_new=df.values,ARD=True,logged=model.logged)
            curve = gm.run()

            # get parameter estimates using predicted curve
            diauxie_dict[idx] = curve.params.pop('df_dx')
            params_latent.loc[idx,:] = curve.params

            if posterior:
                params_sample.loc[idx,:] = curve.sample().posterior

        # summarize diauxie results
        diauxie_df = mergeDiauxieDfs(diauxie_dict)

        if posterior:
            gp_params = params_sample.join(params_latent['diauxie'])
        else:
            gp_params = params_latent

        gp_params = x_min.join(gp_params)
        gp_params.index.name = 'Sample_ID'
        gp_params = gp_params.reset_index(drop=False)
        gp_params = pd.merge(gp_params,diauxie_df,on='Sample_ID')

        # save gp_data fit
        x_out = x_full.copy()
        for key,mapping in factor_dict.items():
            if key in x_out.keys():
                mapping = {str(k):str(v) for k,v in mapping.items()}
                x_out.loc[:,key] = x_out.loc[:,key].replace(reverseDict(mapping)).astype(int)
            if key in gp_params.keys():
                mapping = {str(k):str(v) for k,v in mapping.items()}
                gp_params.loc[:,key] = gp_params.loc[:,key].replace(reverseDict(mapping)).astype(int)

        #params = initParamList(0)
        diauxie = initDiauxieList()
        params = initParamList(0) + initParamList(1)
        params = list(set(params).intersection(set(gp_params.keys())))

        df_params = gp_params.drop(labels=diauxie,axis=1).drop_duplicates()
        df_params = minimizeParameterReport(df_params)
        df_diauxie = gp_params[gp_params.diauxie==1].drop(labels=params,axis=1)
        df_diauxie = minimizeDiauxieReport(df_diauxie)

        # because pooling, drop linear AUC, K, and Death 
        to_remove = ['death_lin','k_lin','auc_lin']
        if do_not_log_transform:
            to_remove += ['td']

        df_params = removeFromParameterReport(df_params,to_remove)
        df_diauxie = removeFromDiauxieReport(df_diauxie,to_remove)

        if do_not_log_transform:
            columns = {'auc_log':'auc_lin','k_log':'k_lin','death_log':'death_lin',
                       'dx_auc_log':'dx_auc_lin','dx_k_log':'dx_k_lin','dx_death_log':'dx_death_lin'}
            df_params.rename(columns=columns,inplace=True)
            df_diauxie.rename(columns=columns,inplace=True)

        if posterior:
            df_params = prettyifyParameterReport(df_params,variable,confidence)
            df_params = articulateParameters(df_params,axis=0)

        summ_path = assembleFullName(dir_path,'',file_name,'params','.txt')
        diux_path = assembleFullName(dir_path,'',file_name,'diauxie','.txt')

        #plate_cond.to_csv(file_path,sep='\t',header=True,index=True)
        df_params.to_csv(summ_path,sep='\t',header=True,index=posterior)
        if df_diauxie.shape[0]>0:
            df_diauxie.to_csv(diux_path,sep='\t',header=True,index=False)

        if save_latent:
            file_path = assembleFullName(dir_path,'',file_name,'output','.txt')
            x_out.to_csv(file_path,sep='\t',header=True,index=True)


    def computeFullDifference(self):
        '''
        Computes the full difference between two latent function (modelling growth curves).

        Args:
            x_diff (pandas.DataFrame): must include columns of Time, mu (mean of latent 
                function), Sigma (diagonal covariance of latent function)
            variable (str): variable of interest, must be a column name in x_diff
            confidence (float [0.0,1.0]): confidence interval, e.g. 0.95 for 95%.
            n (int): number of samples from posterior distribution
            posterior (boolean), whether to sample from posterior distribution
            noise (boolean): whether to plot 95-pct credibel intervals including sample uncertainty

        Returns:
            df (pandas.DataFrame)
            delta_od_sum (float): ||OD(t)||^2 which is defined as the sum of squares 
                for the OD when the mean and its credible interval deviates from zero.
        '''

        x_diff = self.x_full
        variable = self.target[0]
        confidence = self.args.confidence  # confidence interval, e.g. 0.95
        z_value = 1-(1 - confidence)/2
        noise = self.args.include_gaussian_noise
        save_latent = self.args.save_gp_data

        def buildTestMatrix(x_time):
            '''
            Build a test matrix to simlpify OD full difference computation.
                See https://github.com/ptonner/gp_growth_phenotype/testStatistic.py 
                This is used to compare two growth latent functions. The differeence between
                first time points (measurements) are adjusted to zero. 
            Args:
                x_time (pandas.DataFrame or pandas.Series or numpy.ndarray), ndim > 1
            Returns:
                A (numpy.ndarray): N-1 x 2*N where N is length of time.
            '''

            # buildtestmatrix
            n = x_time.shape[0]
            A = np.zeros((n-1,2*n))
            A[:,0] = 1
            A[range(n-1),range(1,n)] = -1
            A[:,n] = -1
            A[range(n-1),n+np.arange(1,n)] = 1

            return A

        x_diff = x_diff.sort_values([variable,'Time']) # do you really need to sort by variable
        x_time = x_diff.Time.drop_duplicates()

        # define mean and covariance of data
        mu = x_diff['mu'].values
        if noise:
            Sigma = np.diag(x_diff['Sigma'] + x_diff['Noise'])
        else:
            Sigma = np.diag(x_diff['Sigma'])

        # define mean and covariance of functional diffeence
        A = buildTestMatrix(x_time)
        m = np.dot(A,mu)
        c = np.dot(A,np.dot(Sigma,A.T))
        mean,std = m,np.sqrt(np.diag(c))

        # sample the curve for the difference between functions, from an MVN distribution
        n = getValue('n_posterior_samples')
        samples = np.random.multivariate_normal(m,c,n)
        
        # compute the sum of functional differences for all sampled curves
        dos = [np.sqrt(np.sum([ii**2 for ii in s])) for s in samples]
        dos_mu, dos_std = np.mean(dos), np.std(dos)

        # compute the confidence interval for the sum of functional differences
        scaler = norm.ppf(z_value) # define confidence interval scaler for MVN predictions
        ci = (dos_mu-scaler*dos_std, dos_mu+scaler*dos_std)

        # compute credible intervals for the curve of the difference
        y_avg = mean
        y_low = y_avg-scaler*std#
        y_upp = y_avg+scaler*std

        # package results
        t = x_time[1:].values
        df = pd.DataFrame([t,y_avg,y_low,y_upp],index=['Time','Avg','Low','Upp']).T

        self.functional_diff = df
        self.delta_od_sum_mean = dos_mu
        self.delta_od_sum_ci = ci
 
        # save gp_data fit
        dir_path = self.paths_dict['dir']
        file_name = self.paths_dict['filename']      
        if save_latent:
            file_path = assembleFullName(dir_path,'',file_name,'func_diff','.txt')
            df.to_csv(file_path,sep='\t',header=True,index=True)


    def plotPredictions(self):
        '''
        Visualizes the model tested by a specific hypothesis given the data.

        Args:
            x_full (pandas.DataFrame)
            x_min (pandas.DataFrame)
            hypotheis (dictionary): keys are str(H0) and str(H1), values are lists of str
            plate (growth.GrowthPlate obj))
            variable (list): variables of interest
            factor_dict (dictionary): mapping of unique values of variables to numerical integers
            subtract_control (boolean): where control sample curves subtracted from treatment sample curves
            file_name (str): 
            directory (str): path where files/figures should be stored
            args_dict (dictionary): must at least include 'nperm', 'nthin', and 'fdr' as keys and their values

        Action:
            saves a plot as PDF file
        '''

        # get necessary attributs
        x_full = self.x_full
        x_min = self.x_min
        factor_dict = self.factor_dict
        variable = self.target[0]
        plate = self.plate

        subtract_control = self.subtract_control
        directory = self.paths_dict['dir'] 
        file_name = self.paths_dict['filename']

        # get and modify user-accessible parameters from config.py
        plot_params = getHypoPlotParams()  # dict
        tick_spacing = plot_params['tick_spacing']
        legend_loc = plot_params['legend']
        fontsize = plot_params['fontsize']

        colors = getValue('hypo_colors')  # list of colors
        confidence = self.args.confidence  # confidence interval, e.g. 0.95
        z_value = 1-(1 - confidence)/2

        noise = self.args.include_gaussian_noise

        if self.args.dont_plot:
            return None

        # grab mapping of integer codes in design matrix to actual variable labels
        varb_codes_map = reverseDict(factor_dict[variable])  # {codes:vlaues}
 
        # set figure aesthetics
        sns.set_style('whitegrid')
        rcParams['font.family'] = 'sans-serif'
        rcParams['font.sans-serif'] = 'Arial'
  
        # initialize grid
        fig,ax = plt.subplots(2,1,figsize=[5,10.5],sharex=False)

        # for each unique value of variable of interest, plot MVN prediction
        list_values = varb_codes_map.items()
        list_values = sorted(list_values, key=operator.itemgetter(1))
        list_colors = colors[0:x_min.shape[0]]

        # plot MVN predictions
        for v_map,color in zip(list_values,list_colors):
            code,label = v_map
            criteria_real = {variable:[label]}
            criteria_mvn = {variable:code}

            ax[0] = addRealPlotLine(ax[0],plate,criteria_real,color,plot_params)
            ax[0] = addMVNPlotLine(ax[0],x_full,criteria_mvn,label,z_value,color,plot_params,noise)
            ax[0].xaxis.set_major_locator(MultipleLocator(tick_spacing))

        # adjust labels and window limits
        ax[0] = setAxesLabels(ax[0],subtract_control,plot_params,logged=self.args.log_transform)

        # if variable has only 2 values and if requested, plot delta OD
        if (len(list_values) != 2) or (self.args.dont_plot_delta_od):
            fig.delaxes(ax[1])
        else: 
            ax[1] = plotDeltaOD(ax[1],self.functional_diff,ylabel=True,xlabel=True,fontsize=fontsize)
            ax[1].xaxis.set_major_locator(MultipleLocator(tick_spacing))
            ax[0].set_xlabel('')

        ax = dynamicWindowAdjustment(ax)

        ## if user did not pass file name for output, use time stamp
        fig_path = assemblePath(directory,file_name,'.pdf')
        plt.subplots_adjust(wspace=0.15,hspace=0.15)
        savePlotWithLegends(ax[0],fig_path,legend_loc,fontsize=fontsize)


    def reportRegression(self):
        '''
        Describes the log Bayes Factor, and the percentile cut-offs for accepting H1 vs H0 based FDR <=20%.
            Results can be reported to stdout.

        Args:
            hypothesis (dictionary): e.g. {'H0':['Time'],'H1':['Time','Substrate']}
            log_BF (float): log Bayes Factor
            dist_log_BF (lsit of floats): log Bayes Factor based on permutation testing (i.e. null distribution)
            FDR (int): false discovery rate cuto-off (%)
            verbose (boolean)

        Returns:
            M1_Pct_Cutoff (float): FDR-based cut-off for accepting alt. model (actual BF must be higher)
            M0_Pct_Cutoff (flaot): FDR-based cut-off for accepting null model (actual BF must be lower)
            log_BF_Pct (float): percentile of actual log Bayes Factor relative to log Bayes Factor null distribution
            msg (str)
        '''

        hypothesis = self.hypothesis
        log_BF = self.log_BF
        dist_log_BF = self.log_BF_null_dist
        fdr = self.args.false_discovery_rate
        verbose = self.args.verbose

        log_BF_Display = prettyNumberDisplay(log_BF)
        dos_mean = prettyNumberDisplay(self.delta_od_sum_mean)
        dos_low = prettyNumberDisplay(self.delta_od_sum_ci[0])
        dos_upp = prettyNumberDisplay(self.delta_od_sum_ci[1])

        if dist_log_BF is None:

            msg = f'Model Tested: {hypothesis}\n\n' 
            msg += f'log Bayes Factor: {log_BF_Display}\n\n'
            msg += f'Functional Difference [95% CI]: {dos_mean} [{dos_low},{dos_upp}]\n'

            self.msg = f'\n{msg}'
            self.M1_Pct_Cutoff = None
            self.M0_Pct_Cutoff = None
            self.log_BF_Pct = None

            smartPrint(msg,verbose)

            return None 

        nperm = int(len(dist_log_BF)+1)

        # The 20% percentile in null distribution, a log BF higher has FDR <=20% that H1 fits data better than H0
        M1_Pct_Cutoff = np.percentile(dist_log_BF,100-fdr)
        M1_Display = prettyNumberDisplay(M1_Pct_Cutoff)

        # The 80% percentile in null distribution, a lo gBF lower has FDR <=20% that H0 fits data better than H1
        M0_Pct_Cutoff = np.percentile(dist_log_BF,fdr)
        M0_Display = prettyNumberDisplay(M0_Pct_Cutoff)

        # Percentile of actual log BF relative to null distribution
        log_BF_Pct = 100 - percentileofscore(dist_log_BF,log_BF) 
        
        msg = f'The following hypothesis was tested on the data:\n{hypothesis}\n\n' 
        msg += f'log Bayes Factor = {log_BF_Display} '
        msg += f'({log_BF_Pct:.1f}-percentile in null distribution based on {nperm} permutations)\n\n'
        msg += f'For P(H1|D) > P(H0|D) and FDR <= {fdr}%, log BF must be > {M1_Display}\n'
        msg += f'For P(H0|D) > P(H1|D) and FDR <= {fdr}%, log BF must be < {M0_Display}\n'
        msg += f'\nThe functional difference [95% CI] is {dos_mean} [{dos_low},{dos_upp}]\n'

        self.M1_Pct_Cutoff = M1_Pct_Cutoff
        self.M0_Pct_Cutoff = M0_Pct_Cutoff
        self.log_BF_Pct = log_BF_Pct
        self.msg = msg

        smartPrint(msg,verbose)


    def exportReport(self):

        def oneLineRport(**kwargs):
            return pd.DataFrame(columns=[0],index=list(kwargs.keys()),data=list(kwargs.values())).T

        if self.args.subtract_control:
            sc_msg = 'Samples were normalized to their respective control samples before analysis.'
        else:
            sc_msg = 'Samples were modeled without controlling for batch effects '
            sc_msg += '(i.e. subtracting the growth of group/batch-specific control samples).'

        nthin = len(np.unique(self.model.x[:,0]))

        msg = 'The following criteria were used to subset data:\n'
        msg += tidyDictPrint(self.params['subset'])
        msg += '\n'
        msg += self.msg
        msg += '\nData Manipulation: Input was reduced to '
        msg += f'{nthin} equidistant time points. {sc_msg}'
        self.msg = msg

        # compact report of results
        report_args = {'Filename':self.paths_dict['filename'],
                      'Subtract_Control':self.args.subtract_control,
                      'Subset':self.params['subset'],
                      'Hypothesis':self.params['hypothesis'],
                      'LL0':self.LL0,
                      'LL1':self.LL1,
                      'Log_BF':self.log_BF,
                      'FDR':self.args.false_discovery_rate,
                      'M1_FDR_cutoff':self.M1_Pct_Cutoff,
                      'M0_FDR_cutoff':self.M0_Pct_Cutoff,
                      'Permuted_log_BF':self.log_BF_null_dist,
                      'Func_Diff_Mean':self.delta_od_sum_mean,
                      'Func_Diff_CI':self.delta_od_sum_ci}

        dir_path = self.paths_dict['dir']
        file_name = self.paths_dict['filename']

        file_path = assembleFullName(dir_path,'',file_name,'log','.txt')
        self.compactReport = oneLineRport(**report_args)
        self.compactReport.to_csv(file_path,sep='\t',header=True,index=None)

        # save report of data
        file_path = assembleFullName(dir_path,'',file_name,'report','.txt')
        fid = open(file_path,'w')
        fid.write(self.msg)
        fid.close()
