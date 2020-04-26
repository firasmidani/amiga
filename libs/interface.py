#!/usr/bin/env python

'''
AMiGA library of functions for parsing commands and arguments.
'''

__author__ = "Firas Said Midani"
__version__ = "0.2.0"
__email__ = "midani@bcm.edu"

# TABLE OF CONTENTS (6 functions)

# parseCommand
# interpretParameters
# initializeParameter
# checkParameterCommand
# checkParameterText
# integerizeDictValues

# parseCommand

import argparse
import os
import re
import sys

from libs.comm import smartPrint, tidyDictPrint, tidyMessage


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
    parser.add_argument('-i','--input',required=True)
    parser.add_argument('-f','--flag',required=False)
    parser.add_argument('-s','--subset',required=False)
    parser.add_argument('-y','--hypothesis',required=False)
    parser.add_argument('-t','--interval',required=False)
    parser.add_argument('-p','--plot',action='store_true',default=False)
    parser.add_argument('-v','--verbose',action='store_true',default=False)
    parser.add_argument('-np','--number-permutations',action='store',type=int,default=20)
    parser.add_argument('-nt','--time-points-skips',action='store',type=int,default=11)
    parser.add_argument('-fdr','--false-discovery-rate',action='store',type=int,default=20)
    parser.add_argument('--merge-summary',action='store_true',default=False)
    parser.add_argument('--normalize-parameters',action='store_true',default=False)
    parser.add_argument('--plot-derivative',action='store_true',default=False)
    parser.add_argument('--only-basic-summary',action='store_true',default=False)
    parser.add_argument('--save-all-data',action='store_true',default=False)
    parser.add_argument('--save-derived-data',action='store_true',default=False)
    parser.add_argument('--save-fitted-data',action='store_true',default=False)
    parser.add_argument('--save-transformed-data',action='store_true',default=False)
    parser.add_argument('--only-print-defaults',action='store_true',default=False)
    parser.add_argument('--perform-substrate-regression',action='store_true',default=False)
    parser.add_argument('--dont-subtract-control',action='store_true',default=False)
    parser.add_argument('-o','--output',required=False)

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
    args_dict['nthin'] = args.time_points_skips
    args_dict['fdr'] = args.false_discovery_rate
    args_dict['merge'] = args.merge_summary
    args_dict['norm'] = args.normalize_parameters
    args_dict['pd'] = args.plot_derivative
    args_dict['obs'] = args.only_basic_summary
    args_dict['sad'] = args.save_all_data
    args_dict['sdd'] = args.save_derived_data
    args_dict['sfd'] = args.save_fitted_data
    args_dict['std'] = args.save_transformed_data
    args_dict['opd'] = args.only_print_defaults
    args_dict['psr'] = args.perform_substrate_regression
    args_dict['sc'] = not args.dont_subtract_control
    args_dict['fout'] = args.output

    # logical argument definitions

    # if subsetting, then merge summary
    if args_dict['subset']:
        args_dict['merge'] = True

    # if save-all-data passed, then
    if args_dict['sad']:
        args_dict['sdd'] = True
        args_dict['sfd'] = True
        args_dict['std'] = True

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
        msg += 'You can adjust these values in libs/config.py. \n\n'
        msg += tidyDictPrint(config)
        sys.exit(msg)
        
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

    return params_dict


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
    if arg is None:
        param_dict = checkParameterText(filepath,sep=sep)

    # else if user provided argument, parse argument for parameters
    elif len(arg)>0:
        param_dict = checkParameterCommand(arg,sep=sep)

    # otherwise, initialize empty dictionary
    else:
        param_dict = {};

    # if argument parameters should be converted to integers (e.g. time interval)
    if integerize:
        return integerizeDictValues(param_dict)
    else:
        return param_dict


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

    if command is None:
        return None

    # strip flanking semicolons and whitespaces then split by semicolons 
    lines = command.strip('; ').split(';');

    # get names of variables (left of semicolons)
    lines_keys = [ii.split(':')[0] for ii in lines]

    # get list of valuse or instances for all variables
    lines_values = [re.split(sep,ii.split(':')[1]) for ii in lines]

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
            lines_dict[key] = values

    return lines_dict 


def integerizeDictValues(dictionary):
    '''
    Converts items in the values of a dictionary into integers. This will work for values
        that are iterable (e.g. list).

    Args:
        dictionary (dict)

    Returns:
        dicionary (dict) where each value is a lis	t of integers

    Example: input {'CD630_PM-1':str(500)} will return {'CD630_PM1-1':int(500)}
    '''

    if dictionary is None:
        return None

    dictionary = {key:[int(vv) for vv in value] for key,value in dictionary.items()}

    return dictionary


