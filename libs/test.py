#!/usr/bin/env python

'''
AMiGA library of functions for testing hypotheses with GP regression.
'''

__author__ = "Firas Said Midani"
__version__ = "0.2.0"
__email__ = "midani@bcm.edu"

# TABLE OF CONTENTS (11 functions)

# testHypothesis
# checkHypothesis
# shouldYouSubstractControl
# updateMappingControls
# prepRegressionPlate
# tidifyRegressionData
# executeRegression
# computeLikliehood
# reportRegression
# plotHypothesis
# performSubstrateRegression


import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import percentileofscore

from libs.comm import tidyDictPrint, tidyMessage, smartPrint, prettyNumberDisplay
from libs.gp import GP
from libs.growth import GrowthPlate
from libs.utils import subsetDf, timeStamp, getValue, getTimeUnits
from libs.trim import annotateMappings, trimMergeMapping, trimMergeData
from libs.org import assembleFullName, assemblePath

def testHypothesis(data_dict,mapping_dict,params_dict,args_dict,directory_dict,subtract_control=True,sys_exit=True,verbose=False):
    '''
    Perform hypothesis testing using Gaussian Process regression, and computes Bayes Factor, only 
        if user passes a hypothesis.

    Args:
        data_dict (dictionary): keys are unique Plate IDs, values are pandas.dataFrams
            each is structured with number of time points (t) x number of samples + 1 (n+1) 
            becasue Time is not an index but rather a column
        mapping_dict (dictionary): keys are unique Plate IDs, values are pandas.dataFrames
            each is structured with number of wells or samples (n) x number of variables (p)
        params_dict (dictionary): must at least include 'hypo' key and its values
        args_dict (dictionary): must at least include 'nperm', 'nthin', and 'fdr' as keys and their values
        directory_dict (dictionary): keys are folder names, values are their paths
        subtract_control (boolean): whether controm sample curves should be subtracted from treatment sample curves
        sys_exit (boolean): whether system should be exited before returning a value to parent caller
        verbose (boolean)

    Returns:
        log_BF (float): log Bayes Factor = log (P(H1|D)/P(H0|D))
        upper (float): the FDR-based cut-off for log BF to support P(H1|D) > P(H0|D)
        lower (float): the FDR-based cut-off for log BF to support P(H0|D) > P(H1|D)

    Actions:
        prints a message that describes the computed Bayes Factor based on user-passed hypothesis and data. 
    '''

    # define testing parameters
    hypothesis = params_dict['hypo']
    nperm = args_dict['nperm']
    nthin = args_dict['nthin']
    fdr = args_dict['fdr']

    if len(hypothesis)==0:
        msg = 'USER ERROR: No hypothesis has been passed to AMiGA via either command-line or text file.\n'
        return None

    if args_dict['fout']:
        file_name = '{}'.format(args_dict['fout'])
    else:
        file_name = 'Hypothesis_Test_{}'.format(timeStamp())

    variable = checkHypothesis(hypothesis)

    # annotate Subset and Flag columns in mapping files, then trim and merge into single dataframe
    mapping_dict = annotateMappings(mapping_dict,params_dict,verbose)
    master_mapping = trimMergeMapping(mapping_dict,verbose) # named index: Sample_ID

    # if you need to subtract control, retrieve relevant control samples
    if subtract_control is True:
        subtract_control = shouldYouSubtractControl(master_mapping,variable)
    master_mapping = updateMappingControls(master_mapping,mapping_dict,to_do=subtract_control)

    # grab all data
    master_mapping = master_mapping.dropna(1)
    master_data = trimMergeData(data_dict,master_mapping,verbose) # unnamed index: row number

    # package, format, and clean data input
    plate = prepRegressionPlate(master_data,master_mapping,subtract_control,nthin)
    ntimepoints = plate.time.shape[0]

    if args_dict['merge']:
 
        # running model on transformed results and recording results
        file_path_key = assembleFullName(directory_dict['models'],'',file_name,'key','.txt')
        file_path_input = assembleFullName(directory_dict['models'],'',file_name,'input','.txt')
        plate.key.to_csv(file_path_key,sep='\t',header=True,index=True)  # save model results
 
        data = tidifyRegressionData(plate,save_path=file_path_input)

    else:

        data = tidifyRegressionData(plate)

    # compute log Bayes Factor and its null distribution 
    log_BF, dist_log_BF = executeRegression(data,hypothesis,nperm)

    upper,lower,percentile,bayes_msg = reportRegression(hypothesis,log_BF,dist_log_BF,FDR=fdr,verbose=verbose)

    if subtract_control:
        sc_msg = 'Samples were normalized to their respective control samples before modelling.'
    else:
        sc_msg = 'Samples were modelled without controlling for batch effects (i.e. normalizing to control samples).'

    if nthin > 1:
        nt_msg = 'Input was reduced to {} time points.'

    msg = 'The following criteria were used to subset data:\n'
    msg += tidyDictPrint(params_dict['subset'])
    msg += '\n'
    msg += bayes_msg
    msg += '\nData Manipulation: Input was reduced to {} time points. {}'.format(ntimepoints,sc_msg)

    # save report of data
    file_path = assembleFullName(directory_dict['models'],'',file_name,'output','.txt')
    fid = open(file_path,'w')
    fid.write(msg)
    fid.close()

    # plot results
    plotHypothesis(data,hypothesis,subtract_control,directory_dict,args_dict)

    # bid user farewell
    if sys_exit:
        msg = 'AMiGA completed your request and wishes you good luck with the analysis!'
        print(tidyMessage(msg))
        sys.exit()

    return log_BF,upper,lower


def checkHypothesis(hypothesis):
    '''
    Verifies that a user provided a hypothesis ant that is meets the following crieteria. The alternative
        hypothesis must have only two variables, one being time.

    Args:
        hypothesis (dictionary): e.g. {'H0':['Time'],'H1':['Time','Substrate']}

    Returns
        (str)
    '''

    if len(hypothesis)==0:
        msg = 'USER ERROR: No hypothesis has been passed to AMiGA via either command-line or text file.\n'
        sys.exit(msg)

    # what is the variable of interest based on hypothesis?
    variables = hypothesis['H1'].copy()
    variables.remove('Time')
    if len(variables) > 1:
        msg = 'USER ERROR: AMiGA can only perform GP regression on a single variable in addition to time.\n'
        msg += 'User has however selected the following variables: {}.\n'.format(hypothesis['H1'])
        sys.exit(msg)

    return variables[0]


def shouldYouSubtractControl(mapping,variable):
    '''
    Checks if control samples must be subtracted from treatment samples for proper hypothesis testing.
        In particular, make sure that the variable of interest is binary (i.e. it has only two possible
        values in the mapping dataframe. This makes sure that GP regression on variable of interest is 
        performing a test on a binary variable.

    Args:
        mapping (pandas.DataFrame): samples (n) by variables (k)
        variable (str): must be one of the column headers for mapping argument

    Returns:
        (boolean)
    '''

    unique_values = mapping.loc[:,variable].unique()
    if len(unique_values) !=2:
        msg = 'USER ERROR: AMiGA can only perform a binary hypothesis test. '
        msg += 'For the variable of interst ({}), '.format(variable)
        msg += 'There should be only 2 possible values. '
        msg += 'These are the current possible values: {}. '.format(unique_values)
        msg += 'If you see less than 2 values, check your hypothesis for typos. '
        msg += 'If you see more than 2 values, try pairwise testing instead.\n'
        sys.exit(msg)

    # subtract control curves if none of the values correspond to a control
    subtract_control = False
    for value in unique_values:
        sub_map = subsetDf(mapping,{variable:[value]})
        sub_map_controls_n = sub_map[sub_map.Control==1].shape[0]
        sub_map_total_n = sub_map.shape[0]
        if sub_map_controls_n == sub_map_total_n:
            return False  # found a value whose samples all correspond to control wells
    else:
        return True
    

def updateMappingControls(master_mapping,mapping_dict,to_do=False):
    '''
    For all samples in master mapping, find relevant controls and add these controls to the master mapping dataframe.

    Args:
        master_mapping (pandas.DataFrame)
        mapping_dict (dictionary)
        to_do (boolean)

    Returns:
        master_mapping (pandas.DataFrame): will have more rows (i.e. samples) than input
    '''

    # check first if you need to do this
    if not to_do:
        return master_mapping

    # find all unique groups
    plate_groups = master_mapping.loc[:,['Plate_ID','Group']].drop_duplicates()
    plate_groups = [tuple(x) for x in plate_groups.values]

    # grab all relevant control samples
    df_controls = []
    for plate_group in plate_groups:
        pid,group = plate_group
        pid_mapping = mapping_dict[pid]
        df_controls.append(subsetDf(pid_mapping,{'Plate_ID':[pid],'Group':[group],'Control':[1]}))

    # re-assemble the master mapping dataframe, including the propercontrols
    df_controls = pd.concat(df_controls)
    master_mapping = pd.concat([master_mapping.copy(),df_controls.copy()],sort=True)
    master_mapping = master_mapping.reset_index(drop=True)
    master_mapping.index.name = 'Sample_ID'
    master_mapping = master_mapping.sort_values(['Plate_ID','Group','Control'])

    return master_mapping


def prepRegressionPlate(data,mapping,subtract_control,thinning_step):
    '''
    Packages data into a growth.GrowthPlate() object and performs a select number of class functions.

    Args:
        data (pandas.DataFrame): t (number of measurements) by n+1 (number of samples + one column for time)
        mapping (pandas.DataFrame): n (number of samples) by p (number of variables)
        subtract_control (boolean)
        thinning_step (int): how many time points to skip between selected time points. 
    '''

    plate = GrowthPlate(data=data,key=mapping)
    plate.convertTimeUnits(input=getTimeUnits('input'),output=getTimeUnits('output'))
    plate.logData()
    plate.subtractBaseline()
    plate.subtractControl(to_do=subtract_control,drop=True)

    plate.thinMeasurements(thinning_step)

    return plate


def tidifyRegressionData(plate,save_path=None):
    '''
    Prepares a single dataframe for running GP regression.

    Args:
        plate (growth.GrowthPlate)
        save_path (str): default is None

    Returns:
        data (pandas.DataFrame): long-format
    '''

    # melt data frame so that each row is a single time measurement
    #   columns include at least 'Sample_ID' (i.e. specific well in a specific plate) and
    #   'Time' and 'OD'. Additioncal column can be explicilty called by user using hypothesis.
    data = (plate.time).join(plate.data)
    data = pd.melt(data,id_vars='Time',var_name='Sample_ID',value_name='OD')
    data = data.merge(plate.key,on='Sample_ID')

    if save_path:
        data.to_csv(save_path,sep='\t',header=True,index=True)
 
    return data


def executeRegression(data,hypothesis,nperm=0):
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

    LL0 = computeLikelihood(data,hypothesis['H0']);
    LL1 = computeLikelihood(data,hypothesis['H1']);
    log_BF = LL1-LL0;

    if nperm==0:
        return log_BF, None

    null_distribution = []
    for rep in range(nperm):
        null_value = computeLikelihood(data,hypothesis['H1'],permute=True);
        null_distribution.append(null_value-LL0)

    return log_BF, null_distribution 


def computeLikelihood(df,variables,permute=False):
    '''
    Computes log-likelihood of a Gaussian Process Regression inference. Permutation is performed 
        by shuffling the values in each variable (e.g. Substrate, PM, but not time or OD) which 
        maintains the true value counts. 

    Args:
        df (pandas.DataFrame): N x p, where N is the nuimember of individual observations (i.e.
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

    # permutation test, if requested
    if permute:
        to_permute = [ii for ii in variables if ii!='Time'][0]
        x_shuffled = np.random.choice(x.loc[:,to_permute],size=x.shape[0],replace=False)
        x.loc[:,to_permute] = x_shuffled

    # build and optimize model, then return maximized log-likelihood
    opt_model = GP(x,y).fit(optimize=True);
    LL = opt_model.log_likelihood()

    return LL


def reportRegression(hypothesis,log_BF,dist_log_BF=None,FDR=20,verbose=False):
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
    '''

    log_BF_Display = prettyNumberDisplay(log_BF)

    if dist_log_BF is None:

        msg = 'Model Tested: {}\n'.format(hypothesis) 
        msg += 'log Bayes Factor: {0:.3f}\n'.format(log_BF_Display)
        smartPrint(msg,verbose)
        
        return None, None, None

    nperm = int(len(dist_log_BF)+1)

    # The 20% percentile in null distribution, a log BF higher has FDR <=20% that H1 fits data better than H0
    M1_Pct_Cutoff = np.percentile(dist_log_BF,100-FDR)
    M1_Display = prettyNumberDisplay(M1_Pct_Cutoff)

    # The 80% percentile in null distribution, a lo gBF lower has FDR <=20% that H0 fits data better than H1
    M0_Pct_Cutoff = np.percentile(dist_log_BF,FDR)
    M0_Display = prettyNumberDisplay(M0_Pct_Cutoff)

    # Percentile of actual log BF relative to null distribution
    log_BF_Pct = 100 - percentileofscore(dist_log_BF,log_BF) 
    
    msg = 'The following hypothesis was tested on the data:\n{}\n\n'.format(hypothesis) 
    msg += 'log Bayes Factor = {} '.format(log_BF_Display)
    msg += '({0:.1f}-percentile in null distribution based on {1} permutations)\n\n'.format(log_BF_Pct,nperm)
    msg += 'For P(H1|D) > P(H0|D) and FDR <= {}%, log BF must be > {}\n'.format(FDR,M1_Display)
    msg += 'For P(H0|D) > P(H1|D) and FDR <= {}%, log BF must be < {}\n'.format(FDR,M0_Display)
    smartPrint(msg,verbose)

    return M1_Pct_Cutoff,M0_Pct_Cutoff,log_BF_Pct,msg


def plotHypothesis(data,hypothesis,subtract_control,directory,args_dict):
    '''
    Visualizes the model tested by a specific hypothesis given the data.

    Args:
        data (pandas.DataFrmae): long format where each row is a sepcific measurement (well- and time-specific)
        hypothesis (dictionary): keys are 'H0' and 'H1', values are lists of variables (must be column headers in data)
        subtract_control (boolean): where control sample curves subtracted from treatment sample curves
        directory (dictionary): keys are folder names, values are their paths
        args_dict (dictionary): must at least include 'nperm', 'nthin', and 'fdr' as keys and their values

    Action:
        saves a plot as PDF file
    '''

    sns.set_style('whitegrid')
    colors = getValue('hypo_colors')

    fig,ax = plt.subplots(figsize=[8,6])

    variable = hypothesis['H1'].copy()
    variable.remove('Time')
    variable = variable[0]
    values = data.loc[:,variable].unique()

    for value,color in zip(values,colors):

        # extract value-specific data
        long_df = subsetDf(data,{variable:[value]})   # long format: time, od, sample_id, ...
        wide_df = pd.pivot(long_df,index='Time',columns='Sample_ID',values='OD')  # wide format: time x sample_id

        # define x domain
        fit_x = pd.DataFrame(wide_df.index).values;

        # fit GP model
        model = GP(x=pd.DataFrame(long_df.Time),y=pd.DataFrame(long_df.OD))
        model.fit(optimize=True)

        # get quantiles
        low,mid,upp = model.predict_quantiles(fit_x,quantiles=(2.5,50,97.5))
        low,mi,upp = [np.ravel(ii) for ii in [low,mid,upp]]

        # plot actual data, and GP fit
        ax.plot(wide_df,color=color,alpha=0.5,lw=1)
        ax.plot(fit_x,mid,color=color,alpha=1.0,lw=3,label=value)
        ax.fill_between(np.ravel(fit_x),low,upp,color=color,alpha=0.10)
 
    # plot aesthetics
    if subtract_control:
        ylabel = 'Normalized {}'.format(getValue('grid_plot_y_label'))
    else:
        ylabel = getValue('grid_plot_y_label')

    ax.set_xlabel('Time ({})'.format(getTimeUnits('output')),fontsize=20)
    ax.set_ylabel(ylabel,fontsize=20)

    ax.legend(fontsize=20)

    [ii.set(fontsize=20) for ii in ax.get_xticklabels()+ax.get_yticklabels()]
   
    if args_dict['fout']:
        fig_name = args_dict['fout']
    else:
        fig_name = 'Hypothesis_Test_{}'.format(timeStamp())

    fig_path = assemblePath(directory['models'],fig_name,'.pdf')

    plt.subplots_adjust(left=0.15) 
    plt.savefig(fig_path)


def performSubstrateRegresssion(to_do,plate,args,directory):
    '''
    Performs a hypothesis test for each substrate (i.e. comapres to negative control and
        computes Bayes Factor; see testHypothesis for more details). 

    Args:
        to_do (boolean): whether to run internal code or not
        plate (growth.GrwothPlate() object)
        args (dictionary): keys are arguments and value are user/default choices
        directory (dictionary): keys are folder names, values are their paths        

    Action:
        returns plate (object) as is or updates with four additional column variables.
    '''

    if not args['psr']:
        return plate

    bayes = []

    for substrate in plate.key.Substrate.unique():

        # initialize hypothesis test parameters
        args_dict = {'Substrate':['Negative Control',substrate]}
        hypo_param = {'hypo':{'H0':['Time'],'H1':['Time','Substrate']}}

        # format data needed for hypothesis test
        hypo_plate = plate.extractGrowthData(args_dict)
        hypo_plate.subtractControl() 
        hypo_data = hypo_plate.time.join(hypo_plate.data)
        hypo_key = hypo_plate.key

        # and boom goes the dynamite
        print((pid,substrate))
        bf,bf_upper,bf_lower = testHypothesis(hypo_data,hypo_key,hypo_param,args,directory,False,False,args['verbose'])
        bayes.append((substrate,bf,bf_upper,bf_lower))

    # add hypothesis testing results to object's key
    bayes = pd.DataFrame(bayes,columns=['Substrate','log_BF','log_BF_upper','log_BF_lower'])
    plate.key = pd.merge(plate.key,bayes,on='Substrate',how='left')

    return plate

