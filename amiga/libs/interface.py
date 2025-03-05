#!/usr/bin/env python

'''
AMiGA library of functions for parsing commands and arguments.
'''

__author__ = "Firas S Midani"
__email__ = "midani@bcm.edu"


# TABLE OF CONTENTS (5 functions)

# interpretParameters
# initializeParameter
# checkParameterCommand
# checkParameterText
# integerizeDictValues

import os
import re

from .comm import smartPrint, tidyDictPrint


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
        ('hypothesis',r'\+|,',False)
    ]
    
    # initialize all parameters based on their settings
    params_dict = {}
    for pp,sep,integerize in params_settings:
        params_dict[pp] = initializeParameter(files[pp],getattr(args, pp),sep=sep,integerize=integerize)

    smartPrint(tidyDictPrint(params_dict),verbose)

    # if user requests any subsetting, summary results must be merged
    if params_dict['subset'] and (not args.merge_summary):

        args.merges = True

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
    if arg is None:
        param_dict = checkParameterText(filepath,sep=sep)

    # else if user provided argument, parse argument for parameters
    elif len(arg)>0 and arg.isdigit():
        param_dict = float(arg)

    elif len(arg)>0:
        param_dict = checkParameterCommand(arg,sep=sep)

    # otherwise, initialize empty dictionary
    else:
        param_dict = {}

    # if argument parameters should be converted to integers (e.g. time interval)
    if integerize and isinstance(param_dict,dict): 
        return integerizeDictValues(param_dict)
    else: 
        return param_dict

def get_sep(sep):
        '''
        Ensures safe handling of non-regex separators by re.split()

        Args:
            sep (str): should be either ',' or '\+|,'
        
        Returns:
            (str)
        '''
        return sep if '|' in sep else re.escape(sep)

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
    lines = command.strip('; ').split(';')

    # get names of variables (left of semicolons)
    lines_keys = [ii.split(':')[0].strip() for ii in lines]

    # get list of valuse or instances for all variables
    lines_values = [[jj.strip() for jj in re.split(get_sep(sep), ii.split(':')[1])] for ii in lines]

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
        fid = open(filepath)
        for line in fid:
            key,value = line.split(':')
            values = re.split(get_sep(sep), value.strip('\n'))
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

    if dictionary is None:
        return None

    for key,value in dictionary.items():

        if isinstance(value,list):
            dictionary[key] = float(value[0])
        else:
            dictionary[key] = float(value)

    return dictionary

