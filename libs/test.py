#!/usr/bin/env python

'''
AMiGA library of functions for object-oriented testing of hypotheses with GP regression.
'''

__author__ = "Firas S Midani"
__email__ = "midani@bcm.edu"

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys

from scipy.stats import norm, percentileofscore
from matplotlib import rcParams
from matplotlib.ticker import MultipleLocator
from tabulate import tabulate
from functools import reduce

from libs.detail import updateMappingControls, shouldYouSubtractControl
from libs.model import GrowthModel
from libs.growth import GrowthPlate
from libs.org import assembleFullName, assemblePath
from libs.trim import annotateMappings,trimMergeMapping,trimMergeMapping, trimMergeData
from libs.utils import subsetDf, reverseDict
from libs.utils import getValue, getTimeUnits, getHypoPlotParams, selectFileName
from libs.comm import *
from libs.params import * 
from libs.plot import * 


class HypothesisTest(object):

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
        self.params = params_dict # need keys 'hypo' and 'subset' 
        self.directory = directory_dict
        self.hypothesis = params_dict['hypo']
        self.subtract_control = self.args['sc']
        self.verbose = self.args['verbose']

        # only proceed if hypothesis is valid
        if self.checkHypothesis(): return None

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
        self.plotPredictions()
        self.reportRegression()
        self.exportReport()

        # bid user farewell
        if sys_exit: sys.exit(tidyMessage('AMiGA completed your request!'))


    def initPaths(self):
        '''
        Initialize paths for for saving data and results. 
        '''

        # if user did not pass file name for output, use time stamp
        file_name = selectFileName(self.args['fout'])
        dir_path = assemblePath(self.directory['models'],file_name,'')
        if not os.path.exists(dir_path): os.mkdir(dir_path)      

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
            msg = 'USER WARNING: Your null and alternaive hypotheses differ by more than one variables. '
            msg += 'All of these distinct variables will be permuted for credible interval testing.\n'
            smartPrint(msg,True)

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
        variables = set([i for value in hypothesis.values() for i in value])
        variables = variables.difference(set(['Time']))  # remove Time
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
                    df.loc[:,variable] = df.apply(lambda x: '{} x {}'.format(x[pairs[0]],x[pairs[1]]),axis=1) 

                    mapping = pd.merge(mapping.reset_index(),df,on=pairs,how='left')
                    mapping = mapping.set_index('Sample_ID')


    def checkControlSamples(self,mapping_dict=None):

        mm = self.master_mapping
        sc = self.subtract_control 

        if sc: sc = shouldYouSubtractControl(mm,self.target)
        
        mm = updateMappingControls(mm,mapping_dict,to_do=sc).dropna(1)
        

    def defineData(self,data_dict=None):

        # grab all data
        self.master_data = trimMergeData(data_dict,self.master_mapping,self.args['nskip'],self.verbose) # unnamed index: row number

    def prettyTabulateSamples(self):

        if self.verbose:
            print_map = self.master_mapping.copy().drop(['Flag','Subset'],axis=1)
            tab = tabulate(print_map,headers='keys',tablefmt='psql')
            msg = 'The following samples will be used in hypothesis testing:'
            msg = '\n{}\n{}\n'.format(msg,tab)
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
        plate.logData()
        plate.subtractBaseline(to_do=True,poly=getValue('PolyFit'),groupby=list(self.non_time_varbs))
        plate.subtractControl(to_do=self.subtract_control,drop=True)
        plate.key.to_csv(self.paths_dict['key'],sep='\t',header=True,index=True)  # save model results
        #plate.thinMeasurements(thinning_step)

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

        nthin = self.args['nthin']
        save_path = self.paths_dict['input']

        variables = self.variables
        non_times = self.non_time_varbs

        # melt data frame so that each row is a single time measurement
        #   columns include at least 'Sample_ID' (i.e. specific well in a specific plate) and
        #   'Time' and 'OD'. Additioncal column can be explicilty called by user using hypothesis.

        data = (self.plate.time).join(self.plate.data)
        data = pd.melt(data,id_vars='Time',var_name='Sample_ID',value_name='OD')
        data = data.merge(self.plate.key,on='Sample_ID')

        if save_path: data.to_csv(save_path,sep='\t',header=True,index=True)

        # reduce dimensionality
        data = data.reindex(['OD']+variables,axis=1)
        data = data.sort_values('Time').reset_index(drop=True) # I don't think that sorting matters, but why not
        self.data = data

        value_counts = data.loc[:,non_times].apply(lambda x: len(np.unique(x)),axis=0)
        if any(value_counts >2):
            value_counts = pd.DataFrame(value_counts).reset_index()
            value_counts.columns = ['Variable','# Unique Values']
            tab = tabulate(value_counts,headers='keys',tablefmt='psql')

            msg = '\n USER WARNING: Hypothesis testing should only be performed on binary conditions.'
            msg = '{} See below.\n\n{}\n'.format(msg,tab)
            sys.exit(msg)


    def factorizeCategoicals(self):

        data = self.data

        # factorize variable values and store mapping
        self.factor_dict = {}
        for varb in self.variables:
            if varb == 'Time': continue
            values = data[varb].unique()
            values_code = range(len(values))
            values_maps = {v:c for v,c in zip(values,values_code)}
            data.loc[:,varb] = data.loc[:,varb].replace(values_maps)
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
        fix_noise = self.args['fn']
        nperm = self.args['nperm']
        nthin = self.args['nthin']

        data = self.data

        data0 = data.loc[:,['OD']+hypothesis['H0']]
        data1 = data.loc[:,['OD']+hypothesis['H1']]

        gm0 = GrowthModel(df=data0,ARD=True,heteroscedastic=fix_noise,nthin=nthin)
        gm1 = GrowthModel(df=data1,ARD=True,heteroscedastic=fix_noise,nthin=nthin)

        gm0,LL0 = gm0.run(predict=False)
        gm1,LL1 = gm1.run(predict=False)
        log_BF = LL1-LL0;

        self.log_BF = log_BF
        self.model = gm1
        self.LL0 = LL0
        self.LL1 = LL1
        self.log_BF_null_dist = None

        null_distribution = []
        to_permute = list(set(hypothesis['H1']).difference(set(hypothesis['H0'])))[0]
        for rep in range(nperm):
            smartPrint('Permutation #{}'.format(rep),verbose)
            null_distribution.append(gm1.permute(to_permute)-LL0)
        smartPrint('',verbose)
        if null_distribution:  self.log_BF_null_dist = null_distribution


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
        fix_noise = self.args['fn']

        # first, generates a dataframe where each row is a unique permutations of non-time  variables
        x = data.loc[:,model_input].drop(['Time'],axis=1)
        ncols = x.shape[1]
        x = x.drop_duplicates().reset_index(drop=True)
        x_min = x.copy()
        x['merge_key'] = [1]*x.shape[0]

        # second, sample evenly spaced time points based on min and max time points
        x_time = pd.DataFrame(data.Time.unique(),columns=['Time'])
        x_time['merge_key'] = [1]*x_time.shape[0]

        # combine time and variable permutation dataframes, such that each unique time point 
        #    is mapped to all unique permutations of variabels
        x = pd.merge(x_time,x,on='merge_key').drop(['merge_key'],axis=1)

        if fix_noise: sigma_noise =np.ravel(model.error_new)+model.noise
        else: sigma_noise = np.ravel([model.noise]*x.shape[0])

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

        data = self.data
        model = self.model
        hypothesis = self.hypothesis
        factor_dict = self.factor_dict

        posterior = self.args['slf']
        save_latent = self.args['sgd']
        fix_noise = self.args['fn']

        dir_path = self.paths_dict['dir']
        file_name = self.paths_dict['filename']

        # get user-defined parameters from config.py 
        dx_ratio_min = getValue('diauxie_ratio_min')
        dx_ratio_varb = getValue('diauxie_ratio_varb')
        scale = getValue('params_scale')
        posterior_n = getValue('n_posterior_samples')

        # define hypothesis paraameters
        model_input = hypothesis['H1']  #grab minimal input data for prediction
        x_full = self.x_full
        x_min = self.x_min

        diauxie_dict = {}
        params_latent = initParamDf(x_min.index,0)

        for idx,row in x_min.iterrows():

            # get x and y data
            df = subsetDf(x_full.drop(['mu','Sigma','Noise'],1),row.to_dict())

            # get curve based on model predictions
            gm = GrowthModel(model=model.model,x_new=df.values,ARD=True)
            curve = gm.run()

            # get parameter estimates using predicted curve
            diauxie_dict[idx] = curve.params.pop('df_dx')
            params_latent.loc[idx,:] = curve.params

        diauxie_df = mergeDiauxieDfs(diauxie_dict)
        gp_params = x_min.join(params_latent)
        gp_params.index.name = 'Sample_ID'
        gp_params = gp_params.reset_index(drop=False)
        gp_params = pd.merge(gp_params,diauxie_df,on='Sample_ID')

        # save gp_data fit
        x_out = x_full.copy()
        for key,mapping in factor_dict.items():
            if key in x_out.keys():
                x_out.loc[:,key] = x_out.loc[:,key].replace(reverseDict(mapping))
            if key in gp_params.keys():
                gp_params.loc[:,key] = gp_params.loc[:,key].replace(reverseDict(mapping))

        params = initParamList(0)
        diauxie = initDiauxieList()

        df_params = gp_params.drop(diauxie,axis=1).drop_duplicates()
        df_params = minimizeParameterReport(df_params)
        df_diauxie = gp_params[gp_params.diauxie==1].drop(params,axis=1)
        df_diauxie = minimizeDiauxieReport(df_diauxie)

        summ_path = assembleFullName(dir_path,'',file_name,'params','.txt')
        diux_path = assembleFullName(dir_path,'',file_name,'diauxie','.txt')

        #plate_cond.to_csv(file_path,sep='\t',header=True,index=True)
        df_params.to_csv(summ_path,sep='\t',header=True,index=False)
        if df_diauxie.shape[0]>0:
            df_diauxie.to_csv(diux_path,sep='\t',header=True,index=False)

        if save_latent:
            file_path = assembleFullName(dir_path,'',file_name,'output','.txt')
            x_out.to_csv(file_path,sep='\t',header=True,index=True)


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
        hypothesis = self.hypothesis
        variables = self.target
        plate = self.plate

        subtract_control = self.subtract_control
        directory = self.paths_dict['dir'] 
        file_name = self.paths_dict['filename']

        # get and modify user-accessible parameters from config.py
        plot_params = getHypoPlotParams()  # dict
        tick_spacing = plot_params['tick_spacing']
        legend_loc = plot_params['legend']
        fontsize = plot_params['fontsize']

        posterior_n = getValue('n_posterior_samples')
        colors = getValue('hypo_colors')  # list of colors
        confidence = getValue('confidence')  # confidence interval, e.g. 0.95
        confidence = 1-(1 - confidence)/2

        posterior = self.args['slf']
        noise = self.args['noise']

        if self.args['dp']: return None

        # only plot if certain conditions are met
        if len(variables) > 1:
            msg = 'USER WARNING: The null and alternative hypotheses differed by more '
            msg += 'than one variable. AMiGA is unable to plot a summary of the data '
            msg += 'that visually discriminates growth curves based on more than one variable.\n'
            smartPrint(msg,True)
            return None
        elif len(variables) == 1: variable = variables[0]
        else: return None

        # grab mapping of integer codes in design matrix to actual variable labels
        varb_codes_map = reverseDict(factor_dict[variable])  # {codes:vlaues}
        cond_variables = list(set(hypothesis['H1']).difference(set(['Time',variable])))  # conditioning variables

        # set figure aesthetics
        sns.set_style('whitegrid')
        rcParams['font.family'] = 'sans-serif'
        rcParams['font.sans-serif'] = 'Arial'

        if len(cond_variables)>0:

            if '*' in variable: # if target variable is an interaction term
                x_min_copy = x_min.drop([variable],1)
                x_full_copy = x_full.drop([variable],1)
            else:
                x_min_copy = x_min.copy()
                x_full_copy = x_full.copy()

            for cond_varb in cond_variables:

                fig,ax = plt.subplots(2,2, figsize=[10.5,10.5],sharey=False,sharex=False)

                # split data by non-conditioning variable 
                splitby = [ii for ii in cond_variables if ii!=cond_varb]
                if len(splitby)==0: splitby = variable
                else: splitby = splitby[0]

                cond_varb_values = x_full[cond_varb].unique()
                cond_codes_map = reverseDict(factor_dict[cond_varb])

                for cv,cond_value in enumerate(cond_varb_values):

                    cond_label = cond_codes_map[cond_value]

                    x_split_values = x_full_copy[splitby].unique()
                    x_cond = subsetDf(x_full,{cond_varb:[cond_value]})

                    for xsv in x_split_values:

                        # what are unique vlaues of non-conditioned variable?
                        color = colors[xsv]  # assign color
                        label = reverseDict(factor_dict[splitby])[xsv]  # assign label

                        criteria_real={cond_varb:[cond_label],splitby:[label]}
                        criteria_mvn={cond_varb:[cond_value],splitby:[xsv]}

                        ax[0,cv] = addRealPlotLine(ax[0,cv],plate,criteria_real,color,plot_params)
                        ax[0,cv] = addMVNPlotLine(ax[0,cv],x_cond,criteria_mvn,label,confidence,color,plot_params,noise)
                        ax[0,cv].set_title(cond_label,fontsize=fontsize)
                        xmin,xmax = ax[0,cv].get_xlim()
                        ax[0,cv].xaxis.set_major_locator(MultipleLocator(tick_spacing))

                    # if conditional variable has only 2 values and if requested, plot delta OD
                    if (self.args['pdo']):
                        df,_ = computeFullDifference(x_cond,splitby,confidence,posterior,posterior_n,noise)
                        ax[1,cv] = plotDeltaOD(ax[1,cv],df,ylabel=False,xlabel=True,fontsize=fontsize)
                        ax[1,cv].xaxis.set_major_locator(MultipleLocator(tick_spacing))

                # edit figure labels and window limits
                ax[0,0] = setAxesLabels(ax[0,0],subtract_control,plot_params)
                ax[0,0].set_xlabel('')
                ax[1,0].set_ylabel(r'${\Delta}$(ln OD)',fontsize=fontsize)
                ax = dynamicWindowAdjustment(ax)

                # remove ticklabels except for first column of subplots
                for ax_ii in list(ax[0,1:]) + list(ax[1,1:]): plt.setp(ax_ii,yticklabels=[])

                # if conditional variable is not binary or user id not request delta-od, delete row
                if (not self.args['pdo']):
                    for ax_ii in ax[1,:]:
                        fig.delaxes(ax_ii)
                    for ax_ii in ax[0,:]:
                        ax_ii.set_xlabel('Time ({})'.format(getTimeUnits('output')),fontsize=fontsize)

                # final adjustment and save figure to pre-specified path
                fig_path = assemblePath(directory,file_name,'_split_by_{}.pdf'.format(cond_varb))
                plt.subplots_adjust(wspace=0.15,hspace=0.15)
                savePlotWithLegends(ax[0,-1],fig_path,legend_loc,fontsize=fontsize)

            return None

        else:

            # initialize grid
            fig,ax = plt.subplots(2,1,figsize=[5,10.5],sharex=False)

            # for each unique value of variable of interest, plot MVN prediction
            list_values = varb_codes_map.items();
            list_colors = colors[0:x_min.shape[0]]

            # plot MVN predictions
            for v_map,color in zip(list_values,list_colors):
                code,label = v_map
                criteria_real = {variable:[label]}
                criteria_mvn = {variable:code}

                ax[0] = addRealPlotLine(ax[0],plate,criteria_real,color,plot_params)
                ax[0] = addMVNPlotLine(ax[0],x_full,criteria_mvn,label,confidence,color,plot_params,noise)
                ax[0].xaxis.set_major_locator(MultipleLocator(tick_spacing))

            # adjust labels and window limits
            ax[0] = setAxesLabels(ax[0],subtract_control,plot_params)

            # if variable has only 2 values and if requested, plot delta OD
            if (len(list_values) != 2) or (not self.args['pdo']):
                fig.delaxes(ax[1])
                delta_od_sum_1 = None
                delta_od_sum_2 = None
            else:
                df,dos1,dos2 = computeFullDifference(x_full,variable,confidence,posterior,posterior_n,noise)
                ax[1] = plotDeltaOD(ax[1],df,ylabel=True,xlabel=True,fontsize=fontsize)
                ax[1].xaxis.set_major_locator(MultipleLocator(tick_spacing))
                ax[0].set_xlabel('')

            ax = dynamicWindowAdjustment(ax)

            ## if user did not pass file name for output, use time stamp
            fig_path = assemblePath(directory,file_name,'.pdf')
            plt.subplots_adjust(wspace=0.15,hspace=0.15)
            savePlotWithLegends(ax[0],fig_path,legend_loc,fontsize=fontsize)

        self.delta_sum_od = dos1
        self.delta_sum_od_sig = dos2


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
        fdr = self.args['fdr']
        verbose = self.args['verbose']

        log_BF_Display = prettyNumberDisplay(log_BF)

        if dist_log_BF is None:

            msg = 'Model Tested: {}\n\n'.format(hypothesis) 
            msg += 'log Bayes Factor: {}\n'.format(log_BF_Display)
            smartPrint(msg,verbose)

            self.msg = '\n{}'.format(msg)
            self.M1_Pct_Cutoff = None
            self.M0_Pct_Cutoff = None
            self.log_BF_Pct = None

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
        
        msg = 'The following hypothesis was tested on the data:\n{}\n\n'.format(hypothesis) 
        msg += 'log Bayes Factor = {} '.format(log_BF_Display)
        msg += '({0:.1f}-percentile in null distribution based on {1} permutations)\n\n'.format(log_BF_Pct,nperm)
        msg += 'For P(H1|D) > P(H0|D) and FDR <= {}%, log BF must be > {}\n'.format(fdr,M1_Display)
        msg += 'For P(H0|D) > P(H1|D) and FDR <= {}%, log BF must be < {}\n'.format(fdr,M0_Display)

        self.M1_Pct_Cutoff = M1_Pct_Cutoff
        self.M0_Pct_Cutoff = M0_Pct_Cutoff
        self.log_BF_Pct = log_BF_Pct
        self.msg = msg

        smartPrint(msg,verbose)


    def exportReport(self):

        def oneLineRport(**kwargs):
            return pd.DataFrame(columns=[0],index=list(kwargs.keys()),data=list(kwargs.values())).T

        if self.args['sc']:
            sc_msg = 'Samples were normalized to their respective control samples before analysis.'
        else:
            sc_msg = 'Samples were modeled without controlling for batch effects '
            sc_msg += '(i.e. normalizing to group/batch-specific control samples).'

        msg = 'The following criteria were used to subset data:\n'
        msg += tidyDictPrint(self.params['subset'])
        msg += '\n'
        msg += self.msg
        msg += '\nData Manipulation: Input was reduced to '
        msg += '{} time points. {}'.format(self.ntimepoints,sc_msg)
        self.msg = msg

        # compact report of results
        report_args = {'filename':self.paths_dict['filename'],
                      'subtract_control':self.args['sc'],
                      'subset':self.params['subset'],
                      'hypothesis':self.params['hypo'],
                      'LL0':self.LL0,
                      'LL1':self.LL1,
                      'log_BF':self.log_BF,
                      'FDR':self.args['fdr'],
                      'upper':self.M1_Pct_Cutoff,
                      'lower':self.M0_Pct_Cutoff,
                      'perm_log_BF':self.log_BF_null_dist,
                      'delta_od_sum':self.delta_sum_od,
                      'delta_od_sum_sig':self.delta_sum_od_sig}

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


def computeFullDifference(x_diff,variable,confidence,posterior,n,noise=False):
    '''
    Computes the full difference between two latent function (modelling growth curves).

    Args:
        x_diff (pandas.DataFrame): must include columns of Time, mu (mean of latent 
            function), Sigma (diagonal covariance of latent function)
        variable (str): variable of interest, must be a column name in x_diff
        confidence (float [0.0,1.0]): confidence interval, e.g. 0.95 for 95%.
        n (int): number of samples from posterior distribution
        eirorposterior (boolean), whether to sample from posterior distribution
        noise (boolean): whetehr to plot 95-pct credibel intervals including sample uncertainty

    Returns:
        df (pandas.DataFrame)
        delta_od_sum (float): ||OD(t)||^2 which is defined as the sum of squares 
            for the OD when the mean and its credible interval deviates from zero.
    '''

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

    scaler = norm.ppf(confidence) # define confidence interval scaler for MVN predictions

    x_diff = x_diff.sort_values([variable,'Time']) # do you really need to sort by variable
    x_time = x_diff.Time.drop_duplicates()

    # define mean and covariance of data
    mu = x_diff['mu'].values
    if noise: Sigma = np.diag(x_diff['Sigma'] + x_diff['Noise'])
    else: Sigma = np.diag(x_diff['Sigma'])

    # define mean and covariance of functional diffeence
    A = buildTestMatrix(x_time)
    m = np.dot(A,mu)
    c = np.dot(A,np.dot(Sigma,A.T))
    mean,std = m,np.sqrt(np.diag(c))

    # compute credible intervals
    y_avg = mean
    y_low = y_avg-scaler*std#np.sqrt(np.diag(c))
    y_upp = y_avg+scaler*std#np.sqrt(np.diag(c))

    # package results
    t = x_time[1:].values
    df = pd.DataFrame([t,y_avg,y_low,y_upp],index=['Time','Avg','Low','Upp']).T

    # compute ||OD(t)||^2: delta od summ across all time points
    delta_od_sum_1 = np.sqrt(np.sum([ii**2 for ii in y_avg]))

    # compute ||OD(t)||^2: only on time epoints where interval doez not overlap zeor 
    od = []
    for i,m,l,u in zip(t,y_avg,y_low,y_upp): 
        if (m < 0) and (u < 0): od.append(m)
        elif (m > 0) and (l > 0): od.append(m)

    delta_od_sum_2 = np.sqrt(np.sum([ii**2 for ii in od]))

    return df, delta_od_sum_1, delta_od_sum_2

