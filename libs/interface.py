#!/usr/bin/env python

'''
AMiGA library of functions for parsing commands and arguments.
'''

__author__ = "Firas S Midani"
__email__ = "midani@bcm.edu"


# TABLE OF CONTENTS (6 functions)

# parseCommand
# interpretParameters
# initializeParameter
# checkParameterCommand
# checkParameterText
# integerizeDictValues

import argparse
import os
import re
import sys

from libs.comm import smartPrint, tidyDictPrint, tidyMessage
from libs.utils import selectFileName


def parseCommand(config):
    '''
    Interprets the arguments passed by the user to AMiGA. If the verobse argument 
        is set to True, argumentParser will also print a message summarizing the 
        the user-passed command-line arguments.

    Note: Function is AMiGA-specific and should not be used verbatim for other apps.

    Args:
        config (dictionary): variables saved in config.py where key is variable and value is value

    Returns:
        args (dict): a dictionary with keys as suggested variable names
            and keys as the user-passed and argparse-interpreted arguments.
    '''

    args_dict= {};

    # parse arguments 
    parser = argparse.ArgumentParser()

    # defining input/output
    parser.add_argument('-i','--input',required=True)
    parser.add_argument('-o','--output',required=False)

    # reducing data
    parser.add_argument('-f','--flag',required=False)
    parser.add_argument('-s','--subset',required=False)

    # selecting time points
    parser.add_argument('-tss','--time-step-size',action='store',type=int,default=1)#11
    parser.add_argument('-sfn','--skip-first-n',action='store',type=int,default=0)
    parser.add_argument('-t','--interval',required=False)

    # hypothesis testing
    parser.add_argument('-y','--hypothesis',required=False)
    parser.add_argument('-fdr','--false-discovery-rate',action='store',type=int,default=10)
    parser.add_argument('-np','--number-permutations',action='store',type=int,default=0)
    parser.add_argument('--subtract-control',action='store_true',default=False)

    # pooling and normalizations
    parser.add_argument('--normalize-parameters',action='store_true',default=False)
    parser.add_argument('--pool-by',required=False)
    parser.add_argument('--normalize-by',required=False)

    # plotting
    parser.add_argument('--plot',action='store_true',default=False)
    parser.add_argument('--plot-derivative',action='store_true',default=False)
    parser.add_argument('--plot-delta-od',action='store_true',default=True)
    parser.add_argument('--dont-plot',action='store_true',default=False)

    ## saving tables
    parser.add_argument('--save-cleaned-data',action='store_true',default=False)
    parser.add_argument('--save-gp-data',action='store_true',default=False)
    parser.add_argument('--save-mapping-tables',action='store_true',default=False)
    parser.add_argument('--merge-summary',action='store_true',default=False)

    ## model preferences
    parser.add_argument('--fix-noise',action='store_true',default=False)
    parser.add_argument('--sample-posterior',action='store_true',default=False)
    parser.add_argument('--include-gaussian-noise',action='store_true',default=False)

    ## user communication
    parser.add_argument('--only-basic-summary',action='store_true',default=False)
    parser.add_argument('--only-print-defaults',action='store_true',default=False)
    parser.add_argument('-v','--verbose',action='store_true',default=False)


    # pass arguments to local variables 
    args = parser.parse_args()
    args_dict['fpath'] = args.input  # File path provided by user
    args_dict['flag'] = args.flag
    args_dict['subset'] = args.subset
    args_dict['hypo'] = args.hypothesis
    args_dict['interval'] = args.interval
    args_dict['plot'] = args.plot
    args_dict['verbose'] = args.verbose
    args_dict['nperm'] = args.number_permutations
    args_dict['nthin'] = args.time_step_size
    args_dict['nskip'] = args.skip_first_n
    args_dict['fdr'] = args.false_discovery_rate
    args_dict['pool'] = [1 if args.pool_by is not None else 0][0]#: args.pool_replicates
    args_dict['merges'] = args.merge_summary
    args_dict['norm'] = args.normalize_parameters
    args_dict['pd'] = args.plot_derivative
    args_dict['pdo'] = args.plot_delta_od
    args_dict['obs'] = args.only_basic_summary
    args_dict['scd'] = args.save_cleaned_data
    args_dict['sgd'] = args.save_gp_data
    args_dict['smt'] = args.save_mapping_tables
    args_dict['opd'] = args.only_print_defaults
    args_dict['sc'] = args.subtract_control
    args_dict['dp'] = args.dont_plot
    args_dict['fout'] = args.output
    args_dict['pb'] = args.pool_by
    args_dict['nb'] = args.normalize_by
    args_dict['fn'] = args.fix_noise
    args_dict['slf'] = args.sample_posterior
    args_dict['noise'] = args.include_gaussian_noise


    # logical argument definitions

    # if normalizing parameters passed,
    if args_dict['nb']:
        args_dict['norm'] = True

    # if subsetting, then merge summary
    if args_dict['subset']:
        args_dict['merges'] = True

    # summarize command-line artguments and print
    if args_dict['verbose']:
        msg = '\n'
        msg += tidyMessage('User provided the following command-line arguments:')
        msg += '\n' 
        msg += tidyDictPrint(args_dict)
        print(msg)

    # print default settings for select variables if prompted by user
    if args_dict['opd']:
        msg = '\nDefault settings for select variables. '
        msg += 'You can adjust these values in amiga/libs/config.py. \n\n'
        msg += tidyDictPrint(config)
        sys.exit(msg)
        
    # if user requests any subsetting, summary results must be merged
    if args_dict['subset'] is not None:
        args_dict['merges'] = True

    return args_dict


def interpretParameters(files,args,verbose=False):
    '''
    Checks specific directories for their existence and makes sure data was 
        provided by user. It will also compose and can print a message that can
        communicate with user the results of this validation.

    Args:
        files (dict): keys are parameter names and values are file paths
        args (dict): keys are parameter names and values are corresponding command-line arguments
 
    Returns:
        params_dict (dict): dictionary where keys are variables and values instances
    '''

    # params_settings defines parameters for interpreting parameter files or argument calls
    #     
    #     each item is a 3-tuple where:    
    #     1. parameter name which matches both input argument and text file name (str)
    #     2. delimitor that separates values of a variable of the parameter (str)
    #     3. integerize: whether to convert values of a variable to integers (boolean)

    params_settings = [
        ('interval',',',True),
        ('subset',',',False),
        ('flag', ',',False),
        ('hypo','\+|,',False)
    ]
    
    # initialize all parameters based on their settings
    params_dict = {}
    for pp,sep,integerize in params_settings:
        params_dict[pp] = initializeParameter(files[pp],args[pp],sep=sep,integerize=integerize)

    smartPrint(tidyDictPrint(params_dict),verbose)

    # if user requests any subsetting, summary results must be merged
    if params_dict['subset']:

        args['merges'] = True

        msg = 'WARNING: Because user has requested subsetting of data, '
        msg += 'results will be merged into single summary and/or data file.\n'
        smartPrint(msg,verbose)

    return params_dict,args


def initializeParameter(filepath,arg,sep=',',integerize=False):
    '''
    Parses command-line argument and/or corresponding text file to define parameter.

    Args:
        filepath (str): path to parameter text file
        arg (str): the command-line user-passed value to argument/parameter
        sep (str): separator or delimitor used in argument key-values pairs
        integerize (boolean): whether to convert argument values into integers

    Retruns:
        param_dict (dict): dictionary where keys are variables and values instances
    '''

    # if user did not provide argument, check for text file
    if arg is None: param_dict = checkParameterText(filepath,sep=sep)

    # else if user provided argument, parse argument for parameters
    elif len(arg)>0: param_dict = checkParameterCommand(arg,sep=sep)

    # otherwise, initialize empty dictionary
    else: param_dict = {};

    # if argument parameters should be converted to integers (e.g. time interval)
    if integerize: return integerizeDictValues(param_dict)
    else: return param_dict


def checkParameterCommand(command,sep=','):
    '''
    Parses command-line text argument and formats it into a dictionary.
        (1) Text must be a list of at least one item that are separated by 
        semicolons (;). (2) Each item must be a variable name separated from a 
        list of its values with a colon (:). (3) Each list must have at least
        one value separated by commas (,).

    Args:
        command (str): see above for description of required format
        sep (str): delimitor that separates values of a variable

    Returns:
        lines_dict (dict): keys are variables and values are a list of variable instances
    '''

    if command is None: return None

    # strip flanking semicolons and whitespaces then split by semicolons 
    lines = command.strip('; ').split(';');

    # get names of variables (left of semicolons)
    lines_keys = [ii.split(':')[0].strip() for ii in lines]

    # get list of valuse or instances for all variables
    lines_values = [[jj.strip() for jj in re.split(sep,ii.split(':')[1])] for ii in lines]

    # re-package variables and their list of values or instances into a dictionary
    lines_dict = {ii:jj for ii,jj in zip(lines_keys,lines_values)}

    return lines_dict


def checkParameterText(filepath,sep=','):
    '''
    Parses text-based parameter file and formats its content into a dictionary.
        (1) Items must be separated by newlines (\n). (2) Each item must be a 
        variable name separated from a list of its values with a colon':'. (3) 
        Each list must have at least one value separated by commas (,).

    Args:
        sep (str): delimitor that separates values of a variable

    Returns:
        lines_dict (dict): keys are variables and values are a list of variable instances
    '''

    exists = os.path.exists(filepath)

    lines_dict = {}

    if exists:
        fid = open(filepath,'r')
        for line in fid:
            key,value = line.split(':')
            values = value.strip('\n').split(sep)
            values = [ii.strip() for ii in values]
            values = [float(ii) if ii.isdigit() else ii for ii in values]
            lines_dict[key.strip()] = values

    return lines_dict 


def integerizeDictValues(dictionary):
    '''
    Converts items in the values of a dictionary into integers. This will work for values
        that are iterable (e.g. list).

    Args:
        dictionary (dict)

    Returns:
        dicionary (dict) where each value is a list of integers

    Example: input {'CD630_PM1-1':str(500)} will return {'CD630_PM1-1':int(500)}
    '''

    if dictionary is None: return None

    for key,value in dictionary.items():

        if isinstance(value,list): dictionary[key] = float(value[0])
        else: dictionary[key] = float(value)

    return dictionary

