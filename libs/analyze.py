#!/usr/bin/env python

'''
AMiGA library of functions for analyzing growth curves.
'''

__author__ = "Firas Said Midani"
__version__ = "0.2.0"
__email__ = "midani@bcm.edu"

# TABLE OF CONTENTS (7 functions)

# basicSummaryOnly
# runGrowthFitting
# prepDataForFitting
# mergedGrowthFitting
# saveInputData
# saveFittedData
# normalizeParameters

import sys

from libs.comm import smartPrint, tidyMessage
from libs.growth import GrowthPlate
from libs.test import performSubstrateRegresssion
from libs.utils import timeStamp, getTimeUnits, resetNameIndex
from libs.org import assembleFullName, assemblePath


def basicSummaryOnly(data,mapping,directory,args,verbose=False):
    '''
    If user only requested plotting, then for  each data file, perform a basic algebraic summary
        and plot data. Once completed, exit system. Otherwise, return None.
 
    Args:
        data (dictionary): keys are plate IDs and values are pandas.DataFrames with size t x (n+1)
            where t is the number of time-points and n is number of wells (i.e. samples),
            the additional 1 is due to the explicit 'Time' column, index is uninformative.
        mapping (dictionary): keys are plate IDs and values are pandas.DataFrames with size n x (p)
            where is the number of wells (or samples) in plate, and p are the number of variables or
            parameters described in dataframe.
        directory (dictionary): keys are folder names, values are their paths
        args
        verbose (boolean)

    Returns:
        None: if only_plot_plate argument is False. 
    '''

    if not args['obs']:  # if not only_plot_plaes
        return None

    print(tidyMessage('AMiGA is summarizing and plotting data files'))

    for pid,data_df in data.items():

        # define paths where summary and plot will be saved
        key_file_path = assemblePath(directory['summary'],pid,'.txt')
        key_fig_path = assemblePath(directory['figures'],pid,'.pdf')

        # grab plate-specific samples
        #   index should be well IDs but a column Well should also exist
        #   in main.py, annotateMappings() is called which ensures the above is the case
        mapping_df = mapping[pid]
        mapping_df = resetNameIndex(mapping_df,'Well',False)

        # grab plate-specific data
        wells = list(mapping_df.Well.values)
        data_df = data_df.loc[:,['Time']+wells]

        # update plate-specific data with unique Sample Identifiers 
        sample_ids = list(mapping_df.index.values)
        data_df.columns = ['Time'] + sample_ids

        # create GrowthPlate object, perform basic summary
        plate = GrowthPlate(data=data_df,key=mapping_df)
        plate.convertTimeUnits(input=getTimeUnits('input'),output=getTimeUnits('output'))
        plate.computeBasicSummary()
        plate.computeFoldChange(subtract_baseline=True)

        # plot and save as PDF, also save key as TXT
        plate.plot(key_fig_path)
        plate.saveKey(key_file_path)

        smartPrint(pid,verbose=verbose)
 
    smartPrint('\nSee {} for summary text files.'.format(directory['summary']),verbose)
    smartPrint('See {} for figure PDFs.\n'.format(directory['figures']),verbose)

    msg = 'AMiGA completed your request and '
    msg += 'wishes you good luck with the analysis!'
    print(tidyMessage(msg))

    sys.exit()


def runGrowthFitting(data,mapping,directory,args,config,verbose=False):
    '''
    Uses Gaussian Processes to fit growth curves and infer paramters of growth kinetics.  

    Args:
        data (pandas.DataFrame): number of time points (t) x number of variables plus-one (p+1)
            plus-one because Time is not an index but rather a column.
        mapping (pandas.DataFrame): number of wells/samples (n) x number of variables (p)
        directory (dictionary): keys are folder names, values are their paths
        args (dictionary): keys are arguments and value are user/default choices
        config (dictionary): variables saved in config.py where key is variable and value is value
        verbose (boolean)

    Action:
        saves summary text file(s) in summary folder in the parent directory.
        saves figures (PDFs) in figures folder in the parent directory.
    '''

    # if user did not pass file name for output, use time stamp
    if args['fout']:
        ts = args['fout']
    else:
        ts = timeStamp()  # time stamp, for naming unique files related to this operation

    plate = prepDataForFitting(data,mapping,subtract_baseline=True)

    # if merge-summary selected by user, then save a single text file for summary
    if args['merge']:

        mergedGrowthFitting(plate,directory,args,config,ts)

        return None

    # for each plate, get samples and save individual text file for plate-specific summaries
    for pid in plate.key.Plate_ID.unique():

        smartPrint('Fitting {}'.format(pid),verbose)

        # grab plate-specific summary
        sub_plate = plate.extractGrowthData(args_dict={'Plate_ID':pid})

        if args['sfd'] or args['plot'] or args['pd']:
            store = True
        else:
        	store = False

        sub_plate.model(store,
            dx_ratio_min=config['diauxie_peak_ratio'],dx_fc_min=config['diauxie_fc_min'])  # run model 

        # saveing transformed data, if requested by user
        saveInputData(args['std'],sub_plate,directory['derived'],'',pid,'transformed','.txt',False,False)

        # saving model fits [od and d(od)/dt] as plots and/or text files
        saveFitData(sub_plate,args,directory,pid)

        # perform systematic GP regression on substrates against control, if requested by user
        sub_plate = performSubstrateRegresssion(args['psr'],sub_plate,args,directory)

        # normalize parameters, if requested
        df = sub_plate.key
        df = normalizeParameters(args['norm'],df)

        # format name and save
        df_path = assemblePath(directory['summary'],pid,'.txt')
        df.to_csv(df_path,sep='\t',header=True,index=True)

        #endfor


def prepDataForFitting(data,mapping,subtract_baseline=True):
    '''
    Packages data set into a grwoth.GrowthPlate() object and transforms data in preparation for GP fitting.

    Args:
        data (pandas.DataFrame): number of time points (t) x number of variables plus-one (p+1)
            plus-one because Time is not an index but rather a column.
        mapping (pandas.DataFrame): number of wells/samples (n) x number of variables (p)
       
    Returns:
        plate (growth.GrwothPlate() object)
    '''

    # merge data-sets for easier analysis and perform basic summaries and manipulations
    plate = GrowthPlate(data=data,key=mapping)
    plate.computeBasicSummary()
    plate.computeFoldChange(subtract_baseline=subtract_baseline)
    plate.convertTimeUnits(input=getTimeUnits('input'),output=getTimeUnits('output'))
    plate.logData()  # natural-log transform
    plate.subtractBaseline()  # subtract first T0 (or rather divide by first T0)

    return plate


def mergedGrowthFitting(plate,directory,args,config,ts):
    '''
    Analyze all samples as a single grwoth.GrowthPlate() object and therefore record results into 
        a single summary text file.

    Args: 
        plate (growth.GrwothPlate() object)
        directory (dictionary): keys are folder names, values are their paths
        args (dictionary): keys are arguments and value are user/default choices
        config (dictionary): variables saved in config.py where key is variable and value is value
        ts (str): time stamp (used for naming files)        

    Returns:
        None
    '''

    if args['sfd'] or args['plot'] or args['pd']:
        store = True
    else:
    	store = False

    # saveing transformed data, if requested by user
    saveInputData(args['std'],plate,directory['derived'],'transformed',ts,'','.txt',False,False)

    # running model on transformed results and recording results
    plate.model(store,dx_ratio_min=config['diauxie_peak_ratio'],dx_fc_min=config['diauxie_fc_min'])  # run model

    # saving model fits [od and d(od)/dt] as plots and/or text files
    saveFitData(plate,args,directory,ts)

    # normalize parameters, if requested
    df = plate.key
    df = normalizeParameters(args['norm'],df)

    # format name and save
    file_path = assembleFullName(directory['summary'],'summary',ts,'','.txt')
    df.to_csv(file_path,sep='\t',header=True,index=True)  # save model results

    return None


def saveInputData(to_do,plate,folder,prefix,filename,suffix,extension,input_time=False,input_data=False):
    '''
    Saves the content of plate (growth.GrowthPlate() object) as a tab-separated file.

    Args: 
        to_do (boolean): whether to save derived data or not
        plate (growth.GrwothPlate() object)
        folder (str): full path to file
        prefix (str): prefix to filename (precedes underscore)
        filename (str): filename without extension
        suffix (str): suffix to filename (succedes underscore)
        extension (str): file extension (e.g. txt,tsv,pdf)
        input_time (boolean): whether to use input_time attribute or time attribution of plate object
        input_data (boolean): whether to use input_data attribute or data attribution of plate object

    Returns: 
        None
    '''

    if not to_do:
        return None

    file_path = assembleFullName(folder,prefix,filename,suffix,extension)

    if input_time:
        time = plate.input_time
    else:
        time = plate.time

    if input_data:
        data = plate.input_data
    else:
        data = plate.data

    time.join(data).to_csv(file_path,sep='\t',header=True,index=None)


def saveFitData(plate,args,directory,filename):
    '''
    Saves the GP model fits of plate (growth.GrowthPlate() object) as plots and/or tab-separated text files.
     
    Args:
        plate (growth.GrwothPlate() object)
        args (dictionary): keys are arguments and value are user/default choices
        directory (dictionary): keys are folder names, values are their paths
        filename (str): file name

    Returns:
        None
    '''

    if args['plot']:  # plot OD and its GP estimate

        fig_path = assembleFullName(directory['figures'],'',filename,'fit','.pdf')
        plate.plot(fig_path,plot_fit=True,plot_derivative=False)

    if args['pd']:  # plot GP estimate of dOD/dt (i.e. derivative)

            fig_path = assembleFullName(directory['figures'],'',filename,'derivative','.pdf')
            plate.plot(fig_path,plot_fit=False,plot_derivative=True)

    if args['sfd']:

        file_path = assembleFullName(directory['derived'],'',filename,'fit','.txt')
        plate.floored_real_prediction.to_csv(file_path,sep='\t',header=True,index=True)

        file_path = assembleFullName(directory['derived'],'',filename,'derivative','.txt')
        plate.derivative_prediction.to_csv(file_path,sep='\t',header=True,index=True) 

    return None

def normalizeParameters(to_do,df):
    '''
    Normalizes growth parameters to control samples. 

    Args:
        to_do (boolean): whether to run internal code or not
        df (pandas.DataFrame): rows are samples, columns are experimental variables. Must include
            Plate_ID, Group, Control, auc, k, gr, dr, td, lag.

    Returns:
        df (pandas.DataFrame): input but with an additional 6 columns.
    '''

    if not to_do:
        return df
    
    df_orig = df.copy()
    df_orig_keys = df_orig.columns 
    
    params = ['auc','k','gr','dr','td','lag']
    params_norm = ['norm_{}'.format(pp) for pp in params]
    params_keep = ['Group','Control'] + params
    
    df = df.loc[:,['Plate_ID']+params_keep]
    
    for pid in df.Plate_ID.unique():
        
        df_plate = df[df.Plate_ID==pid].loc[:,params_keep]
        
        for group in df_plate.Group.unique():
            
            df_group = df_plate[df_plate.Group == group].astype(float)

            df_group = df_group / df_group[df_group.Control==1].mean()
                        
            df.loc[df_group.index,params_keep] = df_group.loc[:,params_keep]
            
    df = df.loc[:,params]
    df.columns = params_norm
    
    df = df_orig.join(df)
            
    return df     

