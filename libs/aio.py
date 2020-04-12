#!/usr/bin/env python

'''
DESCRIPTION library for input/output commands for AMiGA
'''

__author__ = "Firas Said Midani"
__version__ = "0.1.0"
__email__ = "midani@bcm.edu"

# TABLE OF CONTENTS

# breakdownFilePath
# findPlateReaderFiles
# readPlateReaderFolder
# parseCommand
# checkParameterCommand
# checkParameterText
# checkDirectoryExists
# checkDirectoryNotEmpty
# checkMetaText
# mapDirectories
# mapFiles
# initializeParameter
# interpretParameters
# isFileOrFolder
# printDirectoryContents
# tidyDictPrint
# tidyMessage
# validateDirectories

import argparse
import os
import re
import sys
import numpy as np
import pandas as pd
import tabulate
import string
import datetime
import matplotlib.pyplot as plt
import seaborn as sns

from functools import reduce

from libs import biolog_pm_layout as bpl
from libs import agp,misc,growth

from scipy.stats import percentileofscore

def smartPrint(msg,verbose):
    '''
    Only print if verbose argument is True. 

        Previously, I would concatenate messages inside a function and print once 
        function is completed, if verbose argument is satisfied. But the side 
        effect is that messages are printed in blocks (i.e. not flushing)
        and printing would not execute.

        In incremental printing (which this streamlines), if a function fails at a 
        specific point, this can be inferred based on where incremental printing would
        have been interrupted right after the point of failure.

    Args:
        msg (str)
        verbose (boolean)
    '''

    if verbose:
        print(msg)

def breakDownFilePath(filepath,copydirectory):
    '''
    Breaks down a file path into several components.

    Args:
        filepath (str)
        save_dirname (str): directory where a copy of the file would be stored

    Returns:
        filename (str): basename without path
        filebase (str): basename without path and without extension
        newfilepath (str): filename with path and with extension repalced to .tsv

    Example input: '/Users/firasmidani/RandomFileName.asc' will return
        filename --> RandomFileName.asc
        filebase --> RandomFileName
        newfilepath --> /Users/firasmiani/RandomFileName.tsv  
    '''

    filename = os.path.basename(filepath)
    filebase = ''.join(filename.split('.')[:-1])
    dirname = os.path.dirname(filepath)

    newfilepath = assemblePath(copydirectory,filebase,'.tsv')

    return filename, filebase, newfilepath
    

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
    parser.add_argument('-np','--number-permutations',action='store',type=int,default=10)
    parser.add_argument('-nt','--time-points-skips',action='store',type=int,default=11)
    parser.add_argument('-fdr','--false-discovery-rate',action='store',type=int,default=20)
    parser.add_argument('--merge-summary',action='store_true',default=False)
    parser.add_argument('--plot-derivative',action='store_true',default=False)
    parser.add_argument('--only-basic-summary',action='store_true',default=False)
    parser.add_argument('--save-derived-data',action='store_true',default=False)
    parser.add_argument('--save-fitted-data',action='store_true',default=False)
    parser.add_argument('--save-transformed-data',action='store_true',default=False)
    parser.add_argument('--only-print-defaults',action='store_true',default=False)
    parser.add_argument('--perform-substrate-regression',action='store_true',default=False)
    parser.add_argument('--dont-subtract-control',action='store_true',default=False)

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
    args_dict['pd'] = args.plot_derivative
    args_dict['obs'] = args.only_basic_summary
    args_dict['sdd'] = args.save_derived_data
    args_dict['sfd'] = args.save_fitted_data
    args_dict['std'] = args.save_transformed_data
    args_dict['opd'] = args.only_print_defaults
    args_dict['psr'] = args.perform_substrate_regression
    args_dict['sc'] = not args.dont_subtract_control

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


def checkDirectoryExists(directory,generic_name='directory',
                         initialize=False,sys_exit=False,max_width=25):
    '''
    Returns a logic argument (True=exists) and a message that could be relayed 
        to user in regard to existence status and actions taken.
    
    Args:
        directory (str): path
        generic_name (str): short name of directory (for communication with user only)
        sys_exit (boolean): request premature termination if directory does not exist
        initialize (boolean): if directory does not exist, initialize it
        max_width (int): used for padding width in string formatting  

    Returns:
        arg (boolean): True indicates that directory exists
        msg (str):  message composed for communication with user
    '''

    # check if directory exists
    exists = os.path.exists(directory)

    # perform action based on whether directory existss and input parameters
    if exists:
        arg = True
        msg = '{:.<{width}}{}\n'.format(generic_name,directory,width=max_width)
    elif sys_exit:
        arg = False
        sys.exit('FATAL USER ERROR: {} {} does not exist.\n'.format(generic_name,directory))
    elif initialize:
        os.makedirs(directory)
        arg = True
        msg = 'WARNING: {} did not exist but was created.\n'.format(directory)
    else:
        arg = False
        msg = 'WARNING: {} does not exist.\n'.format(directory)

    return arg,msg


def checkDirectoryNotEmpty(directory,generic_name='Directory'):
    '''
    Checks if a directory has at least one file.

    Args:
        directory (str): path
        generic_name (str): generic name of directory for communcating with user

    Returns:
        arg (boolean): True indicates that directory exists
        msg (str):  message composed for communication with user
    '''

    numFiles = len(os.listdir(directory))

    if numFiles == 0:
        arg = False
        sys.exit('FATAL USER ERROR: {} {} is empty.\n'.format(generic_name,directory))
    else:
        arg = True
        msg = '{} {} has {} files:'.format(generic_name,directory,numFiles)
        msg += '\n\n'
        msg += '\n'.join(printDirectoryContents(directory)) # print one item per line
        msg += '\n' # pad bottom withe new line

    return arg,msg


def checkMetaText(filepath,verbose=False):
    '''
    Parses meta.txt file into a pandas.DataFrame.

    Args:
        filepath (str): path to meta.txt, must be tab-delimited with first column as "Plate_ID" (i.e. file name)
        verbose (boolean)

    Returns:
        df_meta (pandas.DataFrame)
        df_meta_plates (list)

    '''

    exists = os.path.exists(filepath)

    if not exists:
        df_meta = pd.DataFrame
        df_meta_plates = [];
    else:
        df_meta = pd.read_csv(filepath,sep='\t',header=0,index_col=None)

    # which plates were characterized in meta.txt?
    try:
        df_meta_plates = df_meta.Plate_ID.values
    except:
        df_meta_plates = []

    # neatly prints meta.txt to terminal
    if exists:
        tab = tabulate.tabulate(df_meta,headers='keys',tablefmt='psql')
        msg = '{:.<16}{}\n{}\n'.format('Meta-Data is',filepath,tab)
        smartPrint(msg,verbose)
    else:
        smartPrint('No meta.txt file found\n',verbose)

    return df_meta,df_meta_plates


def mapDirectories(parent):
    '''
    Returns a dictionary where keys are AMiGA-relevant folders and 
        values are the corresponding file paths

    Args:
        parent (str): file path

    Returns:
        directory (dict) where keys and values are strings
    '''

    # initialize dictionary where values are file paths
    directory = {}

    # topmost directory is the input parent directory
    directory['parent'] = parent

    # format file paths for sub-directories 
    children = ['data','derived','mapping','summary','parameters','figures']

    for child in children:

        directory[child] = assemblePath(parent,child)

    return directory


def mapFiles(directory):
    '''
    Returns a dictionary where keys are AMiGA-relevant files and
        vlaues are the corresponding file paths
    '''

    files  = {}

    # only single file of interest in 'mapping' sub-directory
    files['meta'] = '{}/{}.txt'.format(directory['mapping'],'meta')

    # format paths for files in the 'parameter' sub-directory
    children = ['flag','hypo','subset','interval']

    for child in children:

        files[child] = assemblePath(directory['parameters'],child,'.txt')

    return files


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


def isFileOrFolder(filepath):
    '''
    Determines if a path points to a file or directory.

    Args:
        filepath (str): file path 
    '''

    isFile = os.path.isfile(filepath)

    if isFile:
        parent = os.path.dirname(os.path.dirname(filepath))
        filename = os.path.basename(filepath)
        return parent,filename
    else:
        parent = filepath
        return parent,None


def findPlateReaderFiles(directory):
    '''
    Recrusivelys searches a directory for all files with specific extensions.

    Args:
        directory (str): path to data directory

    Returns:
        list_files (list): list of of paths to data files
    '''

    # you can modify this to include other extensions, but internal data must still be tab-delimited 
    acceptable_extensions = ('.txt','TXT','tsv','TSV','asc','ASC')

    # recursively walk through a directory, if nested
    list_files = []

    for (dirpath,dirnames,filenames) in os.walk(directory):

        # only keep files with acceptable extensions
        filenames = [ii for ii in filenames if ii.endswith(acceptable_extensions)]

        # compose and store filepaths but avoid double slashes (i.e. //) between directory names
        for filename in filenames:

            list_files.append(assemblePath(dirpath,filename))
  
    return list_files


def findFirstRow(filepath,encoding):
    '''
    Searches for line that begins with a Well ID (defined as a letter followed by digits), 
        determines the number of rows that need to be skipped for reading this line, and
        indicates if index column was not found.

    Args:
        filepath (str)

    Returns:
        count (int): number of lines that need to be skipped to read data for first well
        index_column (0 or None): location of row namess, if found as first character (0) or not found (None) 
    '''

    fid = open(filepath,'r',encoding=encoding)

    count = 0
    for line in fid.readlines():
        line_start = line.strip().split('\t')[0]
        if isWellId(line_start):
            fid.close()
            index_column = 0  # row names are the zero-indexed column
            return count, index_column
        count += 1
    else:
        fid.close()
        count = 0
        index_column = None  # row names were not found
        return count, index_column


def isWellId(item):
    '''
    Checks if argument is string and if it has a letter for first character and digits for remaining characters.

    Args:
        item (string)

    Returns:
        (boolean)
    '''

    if not isinstance(item,str):
        return False

    if len(item) < 2:
        return False

    if (item[0] in string.ascii_uppercase) and (item[1:].isdigit()):
        return True
    
    return False


def listTimePoints(interval,numTimePoints):
    '''
    Constructs a numpy.array of a time series based on time interval length and number of time points.

    Args:
        interval (int or float, latter preferred)
        numTimePoints (int)

    Returns:
        time_series (numpy.array of floats)
    '''

    time_series = np.arange(start=0.0,stop=interval*numTimePoints,step=interval)

    return time_series

def checkFileEncoding(filepath):
    '''
    Identifies the correct file encoding needed for python open() to read a text file. 
        It does this brutely using error-detection. If none of the pre-accepted encondings
        are detected, AMiGA will prematurely terminate with a descriptive message to user. 

    Args:
        filepath (str)

    Returns:
        encodign (str): see limited options below. 
    '''

    # not sure if ASICC is necesary since ASCII is a subset of UTF-8
    for encoding in ['UTF-8','UTF-16','UTF-32','ASCII']:
        try:
            open(filepath,'r',encoding=encoding).readline()
            return encoding
        except UnicodeDecodeError:
            pass

    # exit 
    msg = 'FATAL DATA ERROR: AMiGA cannot read{}. '.format(filepath)
    msg += 'AMiGA can only read data text files encoded with either '
    msg += 'UTF-8, UTF-16, UTF-32, or ASCII.\n'
    sys.exit(msg) 
 

def readPlateReaderData(filepath,interval,copydirectory,save=False):
    '''
    Reads a single file and adjusts it to AMiGA-desired format. The desired format is 
        time point (row) by well (col, where first column is actual time point, so row
        names are arbitray numerical order (1...n). All cells should be floats.

    Args:
        filepath (str)
        interval (int): time interval length
        copydirectory (str): location  
    '''

    # initialize useful derivatives of filepath
    filename,filebase,newfilepath = breakDownFilePath(filepath,copydirectory)

    # make sure file is ASCII- or BOM-encoded
    encoding = checkFileEncoding(filepath)

    # identify number of rows to skip and presence/location of index column (i.e. row names)
    skiprows,index_col = findFirstRow(filepath,encoding=encoding)

    # read tab-delimited data file
    df = pd.read_csv(filepath,sep='\t',header=None,index_col=index_col,skiprows=skiprows,encoding=encoding)

    # explicitly define time series based on data size and time interval 
    df.columns = listTimePoints(interval=interval,numTimePoints=df.shape[1])

    # if index column is absent, create one 
    if index_col == None:
        nrows = df.shape[0]
        df.index = parseWellLayout(order_axis=0).index[0:nrows].values

    # explicilty assign column names 
    df.T.index.name = 'Time'

    # remove columns (time points) with only NA values (sometimes happens in plate reader files)
    df = df.iloc[:,np.where(~df.isna().all(0))[0]]

    # remove rows (smples) with only NA values (happens if there is meta-data in file after measurements)
    df = df.iloc[np.where(~df.T.isna().all(0))[0],:]
    df = df.astype(float)

    # set to following format: time point (row) by well (column)
    df = df.T.reset_index(drop=False) # values are OD (float) except first column is time (float)

    # save derived data copy in proper location
    if save:
        df.to_csv(newfilepath,sep='\t',header=True)  # does it save header index name (i.e. Time)

    return df

def parseWellLayout(order_axis=1):
    '''
    Initializes a pandas.DataFrame where indices (row names) are well IDs (e.g. C8) and
        variables indicate row letter and column number.

    Args:
        order_axis (int): 0 indicates order by row (i.e. A1,A2,A3,...,B1,B2,B3,...) and
            1 indicates order by column (i.e. A1,B1,C1,...,A2,B2,C2,...).

    Returns:
        df (pandas.DataFrame, size 96x2): row names are well ID, and columns include
            row letter (str), and column number (str)
    '''

    # initialize rows = ['A',...,'H'] and cols = [1,...,12]
    rows = list(string.ascii_uppercase[0:8])
    cols = list(int(ii) for ii in range(1,13))

    list_wells = []
    list_cols = []
    list_rows = []

    # append one well at a time to list tht will later be merged into dataframe
    if order_axis == 1:
        for col in cols:
            for row in rows:
                list_wells.append('{}{}'.format(row,col))
                list_rows.append(row)
                list_cols.append(col)
    else:
        for row in rows:
            for col in cols:
                list_wells.append('{}{}'.format(row,col))
                list_rows.append(row)
                list_cols.append(col)

    # assemble dataframe
    df = pd.DataFrame([list_wells,list_rows,list_cols],index=['Well','Row','Column'])
    df = df.T  # transpose to long format 
    df = df.set_index('Well')

    return df


def readPlateReaderFolder(filename,directory,config,interval_dict={},save=False,verbose=False):
    '''
    Finds, reads, and formats all files in a directory to be AMiGA-compatible.

    Args:
        filename (str or None): str indicates user is intersted in a single data file, None otherwise
        directory (dictionary): keys are folder names, values are their paths
        config (dictionary): variables saved in config.py where key is variable and value is value
        save (boolean): 
        interval_dict (dictionary)
        verbose (boolean)
    '''

    folderpath = directory['data']
    copydirectory = directory['derived']

    # user may have passed a specific file or a directory to the input argument
    if filename:
        filepaths = ['{}/{}'.format(folderpath,filename)]
    else:
        filepaths = findPlateReaderFiles(folderpath)
    # either way, filepaths must be an iterable list or array

    # read one data file at a time
    df_dict = {}
    for filepath in sorted(filepaths):
        
        # communicate with user
        smartPrint('Reading {}'.format(filepath),verbose)

        # get extension-free file name and path for derived copy
        _, filebase, newfilepath = breakDownFilePath(filepath,copydirectory=copydirectory)

        # set the interval time
        if filebase in interval_dict.keys():
            plate_interval = interval_dict[filebase][0]
        else:
            plate_interval = config['interval']

        # read and adjust file to format: time by wells where first column is time and rest are ODs
        df = readPlateReaderData(filepath,plate_interval,copydirectory,save=save)
        df_dict[filebase] = df

    smartPrint('',verbose)  # print empty newline, for visual asethetics only

    return df_dict


def assemblePath(directory,filebase,extension=''):
    '''
    Stitches a full path to a file.

    Args:
        directory (str): path to directory
        filename (str): file name with extension
        extension (str): by default '', allows stitching filenames that alreay have extension

    Returns:
        filepath (str)
    '''

    # check if directory ends with slash, used to avoid double slashes ('//')
    sep = ['' if directory[-1]=='/' else '/'][0]
    file_path = '{}{}{}{}'.format(directory,sep,filebase,extension)

    return file_path


def checkPlateIdColumn(df,filebase):
    '''
    Validates that Plate ID is a column in dataframe.

    Args:
        df (pandas.DatamFrame)

    Returns:
        df (pandas.DataFrame) 
    '''

    if 'Plate_ID' not in df.keys():
        df.loc[:,'Plate_ID'] = [filebase]*df.shape[0]

    return df


def isBiologFromMeta(series):
    '''
    Identifies if a data file is a Biolog plate based on its meta-data.

    Args:
        ser (pandas.Series)

    Returns:
        (boolean)
    '''

    if 'PM' not in series.keys():
        return False
    elif grabFirstValueFromDf(series,'PM') in range(1,7): 
        return True
    else:
        return False


def isBiologFromName(filebase):
    '''
    Determines if file name nomenclature indicates Biolog PM plate. Biolog PM nomenclature is
        {Isolate Name}_PM{PM plate number}-{Replicate number}.

    Args:
        filebase (str)

    Returns:
        (boolean) 
    '''

    if '_' in filebase:
        split_filebase = filebase.split('_')
    else:
        return False

    if not split_filebase[1].startswith('PM'):
        return False

    return True


def initMappingDf(filebase,well_ids):
    '''
    Creates a minimalist file mapping (pandas.DataFrame). Index column are Well IDs, and sole
        value column is 'Plate_ID'. 

    Args:
        filebase (str): Plate ID
        well_ids (list of str): list of Well IDs (e.g. A1,A2,...,H11,H12)

    Returns:
        df_mapping (pandas.DataFrame): n x 1 where n is based on length of well_ids argument.
    '''

    data = [filebase]*len(well_ids)

    df_mapping = pd.DataFrame(index=well_ids,columns=['Plate_ID'],data=data)
    df_mapping.index.name = 'Well'

    return df_mapping


def grabFirstValueFromDf(df,key,fillna=None):
    '''
    Grabs the first value in a specific column in either a data frme or a series.

    Args:
        df (pandas.DataFrame or pandas.Series)
        key (str): column key
        fillna (str or None): if value not found, should replace with this argument

    Returns:
        value (data type is flexible)
    '''

    if key in df.keys():
        return df.loc[:,key].iloc[0]
    else:
        return fillna


def parseBiologLayout():
    '''
    Initializes a pandas.DataFrame that maps the location (well and plate number)
        of each sugar in BIOLOG PM plates.

    Retruns:
        df (pandas.DataFrame)
    '''

    biolog_layout = [bpl.Carbon1,bpl.Carbon2,bpl.PhosphorusAndSulfur]
    biolog_layout += [bpl.PeptideNitrogen1,bpl.PeptideNitrogen2,bpl.PeptideNitrogen3]

    index = [str(ii) for ii in range(1,7)]
    keys = parseWellLayout(order_axis=0).index

    biolog_layout_df = pd.DataFrame(biolog_layout,index=index,columns=keys).T
    biolog_layout_df.columns.name = 'PM'

    return biolog_layout_df 

def initSubstrateDf(pmn):
    '''
    Initializes a pandas DataFrame for substrates corresponding to a specific BIOLOG PM plate.

    Args:
        pmn (str or int or float): Phenotypic Microarray plate number, must be between [1,7)
    '''

    sr = parseBiologLayout().loc[:,str(int(pmn))]
    df = pd.DataFrame(sr)
    df.columns = ['Substrate']

    return df


def initKeyFromMeta(series,well_ids):
    '''
    Initializes a mapping data frame using data derived from meta.txt. Here, it will reproduce
        a pandas.Series to match the number of wells in corresponding data file.

    Args:
        sr (pandas.Series): no size limit
        well_ids (list): list of variable data type

    Returns:
        df_meta (pandas.DataFrame): # rows = len(well_ids), # cols = df.shape[1]
    '''

    df_meta = pd.concat([series]*len(well_ids))
    df_meta.index = well_ids
    df_meta.index.name = 'Well'

    return df_meta


def expandBiologMetaData(sr):
    '''
    Builds a pandas.DataFrame that combines file-specific meta-data with info derived from
        BIOLOG PM substrate lay-out. Datafrmae will include columns for Plate ID, Isolate 
        name, PM number, and Replicate number. 

    Args:
        sr (pandas.Series): describes meta-data for a specific file

    Returns:
        df (pandas.DataFrame)
    '''

    pmn = grabFirstValueFromDf(sr,'PM')  # PM plate number
    df_substrates = initSubstrateDf(pmn)  # mapping of wells to substrates
    df_meta = initKeyFromMeta(sr,df_substrates.index)
    df = df_meta.join(df_substrates)

    return df


def initBiologPlateKey(plate_id,simple=False):
    '''
    Initiailzes a mapping dataframe using only Plate ID. Dataframe will have columns
        for Plate_ID, Isolate name, PM number, and Replicate number.

    Args:
        plate_id (str): data file name (i.e. filebase)
        simple (boolean): whether to include replicate number in mapping dataframe or not

    Returns:
        df_mapping (pandas.DataFrame)
    '''

    list_keys = ['Plate_ID','Isolate','PM','Replicate']
    isolate,pmn,rep = parsePlateName(plate_id,simple = False)
    df_meta = pd.DataFrame(index=list_keys,data=[plate_id,isolate,pmn,rep]).T
    df = expandBiologMetaData(df_meta)

    return df


def parsePlateName(plate_id,simple=False):
    '''
    Deconstructs a file name to identify Isolate name, PM plate number, and Replicate number.

    Args:
        plate_id (str)
        simple (boolean): whether to return Replicate number

    Returns:
        isolate (str)
        pmn (int)
        rep (int)
    '''

    isolate = str(plate_id.split('_')[0])
    pmn = int(plate_id.split('PM')[1][0])
    rep = [int(plate_id.split('-')[-1]) if '-' in plate_id else 1][0]

    if simple:
        return isolate,pmns
    else:
        return isolate,pmn,rep


def assembleMappings(data,mapping_path,meta_path,verbose):
    '''
    Creates a master mapping file (or dictionary ?) for all data files in the input argument.
        For each data file, in this particular order, it will first (1) check if an individual
        mapping file exists, (2) if not, check if relevant meta-data is provided in meta.txt
        file, (3) if not, infer if plate is a BIOLOG PM based on its file name, and (4) if all
        fail, create a minimalist mapping file. 

    Args:
        data (dictionary): keys are file names (i.e. filebases or Plate IDs) and values are
            pandas DataFrames where index column (row names) are well IDs.
        mapping_path (str): path to the mapping folder.
        meta_path (str): path to the mapping file.

    Returns:
        df_mapping_dict (dictionary): keys are file names and values are mapping files. 
    '''

    df_mapping_dict = {}

    # list all data files to be analyed
    list_filebases = data.keys()

    # list all potential mapping file paths
    list_mapping_files = [assemblePath(mapping_path,ii,'.txt') for ii in list_filebases]

    # read meta.txt and list all plates described by it
    meta_df, meta_df_plates = checkMetaText(meta_path,verbose=verbose)   

    # assemble mapping for one data file at a time
    for filebase,mapping_file_path in zip(list_filebases,list_mapping_files):

        # what are the row names from the original data file 
        well_ids = data[filebase].columns[1:]  # this may no be A1 ... H12, but most ofen will be

        # see if user provided a mapping file that corresponds to this data file (filebase)
        if os.path.exists(mapping_file_path):

            df_mapping = pd.read_csv(mapping_file_path,sep='\t',header=0,index_col=0)   
            df_mapping = checkPlateIdColumn(df_mapping,filebase) # makes sure Plate_ID is a column

            smartPrint('{:.<30} Reading {}.'.format(filebase,mapping_file_path),verbose)

        # see if user described the file in meta.txt 
        elif filebase in meta_df_plates:

            meta_info = meta_df[meta_df.Plate_ID==filebase]
            msg = '{:.<30} Found meta-data in meta.txt '.format(filebase)

            biolog = isBiologFromMeta(meta_info)  # does meta_df indicate this is a BIOLOG plate

            if biolog:
                df_mapping = expandBiologMetaData(meta_info)
                msg += '& seems to be a BIOLOG PM plate.'
                smartPrint(msg,verbose)
            else:
                df_mapping = initKeyFromMeta(meta_info,well_ids)
                msg += '& does not seem to be a BIOLOG PM plate.'
                smartPrint(msg,verbose)

        elif isBiologFromName(filebase):

            df_mapping = initBiologPlateKey(filebase)
            msg = '{:.<30} Did not find mapping file or meta-data '.format(filebase)
            msg += 'BUT seems to be a BIOLOG PM plate.'
            smartPrint(msg,verbose)

        else:
            df_mapping = initMappingDf(filebase,well_ids) 
            msg = '{:.<30} Did not find mapping file or meta-data '.format(filebase)
            msg += '& does not seem to be a BIOLOG PM plate.'
            smartPrint(msg,verbose)

        df_mapping_dict[filebase] = expandMappingParams(df_mapping)

    #df_mapping = df_mapping.reset_index(drop=False)
    smartPrint('',verbose)

    return df_mapping_dict

def expandMappingParams(df):
    '''
    Expand input data frame to include columns relevant for AMiGA processing of user paramters.
        It will add columns for Group, Control, Flag, and Subset.

    Args:
        df (pandas.DataFrame)

    Returns:
        df (pandas.DataFrame): with four additional columns
    '''

    # plate can be divided into multiple group, each gorup has unique control well(s)
    df.loc[:,'Group'] = [1]*df.shape[0]  # all wells in a BIOLOG PM plte belong to same group
    df.loc[:,'Control'] = [0]*df.shape[0]  # all wells (except) A1 are treatments
    df.loc['A1','Control'] = 1  # A1 is the control well

    # initialize well-specific flag and subset parameters
    df.loc[:,'Flag'] = [0]*df.shape[0]  # by default, no wells are flagged
    df.loc[:,'Subset'] = [1]*df.shape[0]  # by default, all wells are included in analysis

    return df


def flagWells(df,flags,verbose=False):
    '''
    Passes plate-well-specific flags from user into mapping dataframes.

    Args:
        df (pandas.DataFrame) must have Plate_IDs and Well as columns
        flags (dictionary) with Plate_IDs (str) as keys and Wells (stR) as vlaues
        verbose (boolean)

    Returns:
        df (pandas.DataFrame)
    '''

    if (len(flags)==0):
        smartPrint('No wells were flagged.\n',verbose)
        return df

    for plate, wells in flags.items():
        df[plate].loc[wells,'Flag'] = [1]*len(wells)

    smartPrint('The following flags were detected:\n',verbose)
    smartPrint(tidyDictPrint(flags),verbose)

    return df


def subsetWells(df_mapping_dict,criteria,hypothesis,verbose=False):
    '''
    Tag wells that meet user-passed criteria.

    Args:
        df (pandas.DataFrame) must have Plate_IDs and Well as columns
        criteria (dictionary) with mapping variables (str) as keys and accepted instances (str) as values
        hypothesis (dictionary) 
        verbose (boolean)

    Returns:
        df (pandas.DataFrame)
    '''

    if (len(criteria)==0):
        smartPrint('No subsetting was requested.\n',verbose)
        return df_mapping_dict

    for plate_id,mapping_df in df_mapping_dict.items():

        # subsetting on number-based criteria does not match hits due to type mismatch (str vs int/float)
        mapping_df_str = mapping_df.astype(str)

        remove_boolean = ~(mapping_df_str.isin(criteria).sum(1)==len(criteria)).values  # list of booleans
        remove_idx = mapping_df_str.index[remove_boolean]
        mapping_df.loc[remove_idx,'Subset'] = [0]*len(remove_idx)

    smartPrint('The following criteria were used to subset data:\n',verbose)
    smartPrint(tidyDictPrint(criteria),verbose)

    return df_mapping_dict


def annotateMappings(mapping_dict,params_dict,verbose=False):
    '''
    Annotates the mapping data based on user-passed flags and subsetting criteria. In particular,
        it will annotate the Flag and Subset columns. It will also turn Well index into a
        standalone column. 

    Args:
        mapping_dict (dictionary): of mapping files (pandas.DataFrames), keys are file names (str)
        params_dict (dictionary): dictionary where keys are variables and values instances
        verbose (boolean)

    Returns:
        mapping_dict (dictionary): should be equal or smaller in size than input
    '''

    # flag wells that user does not want to analyze
    mapping_dict = flagWells(mapping_dict,params_dict['flag'],verbose=verbose)

    # tag wells that meet user-passed criteria for analysis
    mapping_dict = subsetWells(mapping_dict,params_dict['subset'],params_dict['hypo'],verbose=verbose)

    # make sure that mappings have Well columns
    #   here we assume that mapping_dict values have index of Well IDs, which should be the case
    mapping_dict = {pid:resetNameIndex(df,'Well',False) for pid,df in mapping_dict.items()}

    return mapping_dict


def trimMergeMapping(mapping_dict,verbose=False):
    '''
    Trims and merges mapping dataframes into one master mapping data frame.

    Args:
        mapping (dictionary): keys are plate IDs and values are pandas.DataFrames with size n x (p)
            where is the number of wells (or samples) in plate, and p are the number of variables or
            parameters described in dataframe.
        params (dictionary): must at least include 'subset' and 'flag' keys and their values
        verbose (boolean)

    Returns:
        mapping (pandas.DataFrame): number of wells/samples (n) x number of variables (p)  
    '''

    # merge mapping dataFrames
    #   sort will force shared (inner) keys to the lead and unshared (outer) keys to the caboose
    #   useful because individual mapping files may lack certain columns, some may even be empty 
    master_mapping = pd.concat(mapping_dict.values(),ignore_index=True,join='outer',sort=False)

    # trim mapping based on Subset and Flag columns
    master_mapping = misc.subsetDf(master_mapping,{'Subset':[1],'Flag':[0]})

    # reset_index and set as Sample_ID
    master_mapping = resetNameIndex(master_mapping,'Sample_ID',True)

    return master_mapping


def trimMergeData(data_dict,master_mapping,verbose=False):
    '''
    Trims and merges data dataframes into one master dataframe. 

    Args:
        data (dictionary): keys are plate IDs and values are pandas.DataFrames with size t x (n+1)
            where t is the number of time-points and n is number of wells (i.e. samples),
            the additional 1 is due to the explicit 'Time' column, index is uninformative.
        mapping_df (pandas.DataFrame): number of well/samples (n) x number of variables (p)
        verbose (boolean)

    Returns:
        master_data (pandas.DataFrame): number of time points (t) x number of variables plus-one (p+1)
            plus-one because Time is not an index but rather a column
    '''

    new_data_dict = {}

    for pid,data in data_dict.items():

        # grab plate-specific samples
        mapping_df = master_mapping[master_mapping.Plate_ID==pid]

        # if no plate-specific samples are included in master_mapping, skip
        if mapping_df.shape[0]==0:
            continue

        # grab plate-specific data
        wells = list(mapping_df.Well.values)
        df = data.loc[:,['Time']+wells]

        # update plate-specific data with unique Sample Identifiers 
        sample_ids = list(mapping_df.index.values)
        df.columns = ['Time'] + sample_ids
        df.T.index.name = 'Sample_ID'

        # udpate dictionary
        new_data_dict[pid] = df

    # each data is time (T) x wells (N), will merge all data and keep a single Time column
    #     reduce(fun,seq) applies a function (fun) recursively to all elements in list (seq)
    #     here, reduce will merge the first two dataFrames in data_dict.values(), then it
    #     will take this output and merge it with third dataFrame in list, and so on
    master_data = reduce(lambda ll,rr: pd.merge(ll,rr,on='Time',how='outer'),new_data_dict.values())
    master_data = master_data.sort_values(['Time']).reset_index(drop=True)

    return master_data
    

def trimInput(data_dict,mapping_dict,params_dict,verbose=False):
    '''
    Interprets parameters to reduce mapping and data files to match user-desired criteria.

    Args:
        data (dictionary): keys are plate IDs and values are pandas.DataFrames with size t x (n+1)
            where t is the number of time-points and n is number of wells (i.e. samples),
            the additional 1 is due to the explicit 'Time' column, index is uninformative.
        mapping (dictionary): keys are plate IDs and values are pandas.DataFrames with size n x (p)
            where is the number of wells (or samples) in plate, and p are the number of variables or
            parameters described in dataframe.
        params (dictionary): must at least include 'subset' and 'flag' keys and their values
        verbose (boolean)

    Returns:
        data (dictionary): values may have smaller size than at time of input
        mapping (dictionary): values may have smaller size than at time of input 
    '''

    # annotate Subset and Flag columns in mapping files
    mapping_dict = annotateMappings(mapping_dict,params_dict,verbose)

    # trim and merge into single pandas.DataFrames
    master_mapping = trimMergeMapping(mapping_dict,verbose) # named index: Sample_ID
    master_data = trimMergeData(data_dict,master_mapping,verbose) # unnamed index: row number

    return master_data,master_mapping


def resetNameIndex(df,index_name,new_index=False):
    '''
    Resets and names index of a pandas.DataFrame.

    Args:
        df (pandas.DataFrame): index (row.names), for mapping dataframes, index should be Well IDs (e.g. A1).
        index_name (str): name of index column, to be assigned.
        new_index (boolean): create new index (row number) and drop current, otherwise keep index as column

    Returns:
        mapping_df (pandas.DataFrame): with an additional coulmn with the header 'Well'.
    '''

    if not new_index: 
        df.index.name = index_name
        df.reset_index(drop=False,inplace=True)

    if new_index:
        df.reset_index(drop=True,inplace=True)
        df.index.name = index_name

    return df


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
        df_controls.append(misc.subsetDf(pid_mapping,{'Plate_ID':[pid],'Group':[group],'Control':[1]}))

    # re-assemble the master mapping dataframe, including the propercontrols
    df_controls = pd.concat(df_controls)
    master_mapping = pd.concat([master_mapping.copy(),df_controls.copy()],sort=True)
    master_mapping = master_mapping.reset_index(drop=True)
    master_mapping.index.name = 'Sample_ID'
    master_mapping = master_mapping.sort_values(['Plate_ID','Group','Control'])

    return master_mapping


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
        sub_map = misc.subsetDf(mapping,{variable:[value]})
        sub_map_controls_n = sub_map[sub_map.Control==1].shape[0]
        sub_map_total_n = sub_map.shape[0]
        if sub_map_controls_n == sub_map_total_n:
            return False  # found a value whose samples all correspond to control wells
    else:
        return True
    

def prepRegressionPlate(data,mapping,subtract_control,thinning_step):
    '''
    Packages data into a growth.GrowthPlate() object and performs a select number of class functions.

    Args:
        data (pandas.DataFrame): t (number of measurements) by n+1 (number of samples + one column for time)
        mapping (pandas.DataFrame): n (number of samples) by p (number of variables)
        subtract_control (boolean)
        thinning_step (int): how many time points to skip between selected time points. 
    '''

    plate = growth.GrowthPlate(data=data,key=mapping)
    plate.convertTimeUnits(input=misc.getTimeUnits('input'),output=misc.getTimeUnits('output'))
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
        data.to_csv('/Users/firasmidani/Downloads/20200303_data.txt',sep='\t',header=True,index=True)
 
    return data


def executeRegression(data,hypothesis,nperm=0):
    '''
    Computes the log Bayes Factor and its null distribution (based on permutation tests).

    Args:
        data (pandas.DataFrame): each row is a single measurement (i.e. time point in a well), columns are variables
            and must include 'Time', 'OD'.
        hypothesis (dictionary): keys must be 'H0' and 'H1', values are lists of variables (must match data keys)
        nperm (int): number of permutations to generate null distribution

    Returns:
        log_BF (float): log Bayes Factor = log (P(H1|D)/P(H0|D))
        null_distribution (list of floats): the null distribution for log Bayes Factor where variable of interest
            was permuted for a certain number of times (based on nperm).
    '''

    LL0 = agp.computeLikelihood(data,hypothesis['H0'])
    LL1 = agp.computeLikelihood(data,hypothesis['H1'])
    log_BF = LL1-LL0

    if nperm==0:
        return log_BF, None

    null_distribution = []
    for rep in range(nperm):
        null_value = agp.computeLikelihood(data,hypothesis['H1'],permute=True)
        null_distribution.append(null_value)

    return log_BF, null_distribution 


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

    if dist_log_BF is None:

        msg = 'Model Tested: {}\n'.format(hypothesis) 
        msg += 'log Bayes Factor: {0:.3f}\n'.format(log_BF)
        smartPrint(msg,verbose)
        
        return None, None, None

    # The 20% percentile in null distribution, a log BF higher has FDR <=20% that H1 fits data better than H0
    M1_Pct_Cutoff = np.percentile(dist_log_BF,100-FDR)

    # The 80% percentile in null distribution, a lo gBF lower has FDR <=20% that H0 fits data better than H1
    M0_Pct_Cutoff = np.percentile(dist_log_BF,FDR)

    # Percentile of actual log BF relative to null distribution
    log_BF_Pct = 100 - percentileofscore(dist_log_BF,log_BF) 

    msg = 'Model Tested: {}\n'.format(hypothesis) 
    msg += 'log Bayes Factor: {0:.3f} '.format(log_BF)
    msg += '({0:.1f}-percentile in null distribution)\n'.format(log_BF_Pct)
    msg += 'For P(H1|D) > P(H0|D) and FDR <= {}%, log BF must be > {:.3f}\n'.format(FDR,M1_Pct_Cutoff)
    msg += 'For P(H0|D) > P(H1|D) and FDR <= {}%, log BF must be < {:.3f}\n'.format(FDR,M0_Pct_Cutoff)
    smartPrint(msg,verbose)

    print(M1_Pct_Cutoff)
    print(M0_Pct_Cutoff)
    print(log_BF_Pct)

    return M1_Pct_Cutoff,M0_Pct_Cutoff,log_BF_Pct


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
    data = tidifyRegressionData(plate)
 
    # compute log Bayes Factor and its null distribution 
    log_BF, dist_log_BF = executeRegression(data,hypothesis,nperm)
    upper,lower,percentile = reportRegression(hypothesis,log_BF,dist_log_BF,FDR=fdr,verbose=verbose)

    # plot results
    plotHypothesisTest(data,hypothesis,subtract_control,directory_dict)

    # bid user farewell
    if sys_exit:
        msg = 'AMiGA completed your request and wishes you good luck with the analysis!'
        print(tidyMessage(msg))
        sys.exit()

    return log_BF,upper,lower


def plotHypothesisTest(data,hypothesis,subtract_control,directory):
    '''
    Visualizes the model tested by a specific hypothesis given the data.

    Args:
        data (pandas.DataFrmae): long format where each row is a sepcific measurement (well- and time-specific)
        hypothesis (dictionary): keys are 'H0' and 'H1', values are lists of variables (must be column headers in data)
        subtract_control (boolean): where control sample curves subtracted from treatment sample curves
        directory (dictionary): keys are folder names, values are their paths

    Action:
        saves a plot as PDF file
    '''

    sns.set_style('whitegrid')
    colors = [(1,0,0),(0,0,1)]

    fig,ax = plt.subplots(figsize=[8,6])

    variable = hypothesis['H1'].copy()
    variable.remove('Time')
    variable = variable[0]
    values = data.loc[:,variable].unique()

    for value,color in zip(values,colors):

        # extract value-specific data
        long_df = misc.subsetDf(data,{variable:[value]})   # long format: time, od, sample_id, ...
        wide_df = pd.pivot(long_df,index='Time',columns='Sample_ID',values='OD')  # wide format: time x sample_id

        # fit GP model
        model = agp.GP(x=pd.DataFrame(long_df.Time),y=pd.DataFrame(long_df.OD))
        model = model.fit()

        # define plot values for GP fit
        fit_x = pd.DataFrame(wide_df.index).values
        fit_mu,fit_var = model.predict(fit_x)
        fit_mu = np.ravel(fit_mu)
        fit_var = np.ravel(fit_var)
        fit_low = fit_mu - fit_var
        fit_upp = fit_mu + fit_var

        # plot actual data, and GP fit
        ax.plot(wide_df,color=color,alpha=0.5,lw=1)
        ax.plot(fit_x,fit_mu,color=color,alpha=1.0,lw=3,label=value)
        ax.fill_between(np.ravel(fit_x),fit_low,fit_upp,color=color,alpha=0.10)
 
    # plot aesthetics
    ax.set_xlabel('Time ({})'.format(misc.getTimeUnits('output')),fontsize=20)
    ax.set_ylabel(misc.getText('grid_plot_y_label'),fontsize=20)

    ax.legend(fontsize=20)

    [ii.set(fontsize=20) for ii in ax.get_xticklabels()+ax.get_yticklabels()]
   
    fig_name = 'Hypothesis_Test_{}'.format(timeStamp())
    fig_path = assemblePath(directory['figures'],fig_name,'.pdf')

    plt.subplots_adjust(left=0.15) 
    plt.savefig(fig_path)


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
        plate = growth.GrowthPlate(data=data_df,key=mapping_df)
        plate.convertTimeUnits(input=misc.getTimeUnits('input'),output=misc.getTimeUnits('output'))
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


def timeStamp():
    '''
    Reports the current time in a certain text format.

    Returns:
        ts (str)
    '''

    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    return ts


def assembleFullName(folder,prefix,filename,suffix,extension):
    '''
    Assembles a file name using the arguments into a complete full path. 

    Args:
        folder (str): full path to file
        prefix (str): prefix to filename (precedes underscore)
        filename (str): filename without extension
        suffix (str): suffix to filename (succedes underscore)
        extension (str): file extension (e.g. txt,tsv,pdf)

    Returns:
        file_path (str): full path to generated file name
    '''

    file_name = '{}_{}_{}'.format(prefix,filename,suffix)
    file_name = file_name.strip('_') # in case prefix is empty

    file_path = assemblePath(folder,file_name,extension)

    return file_path


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
    plate = growth.GrowthPlate(data=data,key=mapping)
    plate.computeBasicSummary()
    plate.computeFoldChange(subtract_baseline=subtract_baseline)
    plate.convertTimeUnits(input=misc.getTimeUnits('input'),output=misc.getTimeUnits('output'))
    plate.logData()  # natural-log transform
    plate.subtractBaseline()  # subtract first T0 (or rather divide by first T0)

    return plate

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

    # saveing transformed data, if requested by user
    saveInputData(args['std'],plate,directory['derived'],'transformed',ts,'','.txt',False,False)

    # running model on transformed results and recording results
    file_path = assembleFullName(directory['summary'],'summary',ts,'','.txt')
    plate.model(diauxie_thresh=config['diauxie'])  # run model
    plate.key.to_csv(file_path,sep='\t',header=True,index=True)  # save model results

    return None


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
        plate.plot(fig_path,plot_fit=True)

    if args['pd']:  # plot GP estimate of dOD/dt (i.e. derivative)

            fig_path = assembleFullName(directory['figures'],'',filename,'derivative','.pdf')
            plate.plot(fig_path,plot_fit=False,plot_derivative=True)

    if args['sfd']:

        file_path = assembleFullName(directory['derived'],'',filename,'fit','.txt')
        plate.prediction.to_csv(file_path,sep='\t',header=True,index=True)

        file_path = assembleFullName(directory['derived'],'',filename,'derivative','.txt')
        plate.derivative_prediction.to_csv(file_path,sep='\t',header=True,index=True) 

    return None


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
        sub_plate.model(args['plot'],diauxie_thresh=config['diauxie'])  # run model 

        # saveing transformed data, if requested by user
        saveInputData(args['std'],sub_plate,directory['derived'],'',pid,'transformed','.txt',False,False)

        # saving model fits [od and d(od)/dt] as plots and/or text files
        saveFitData(sub_plate,args,directory,pid)

        # perform systematic GP regression on substrates against control, if requested by user
        sub_plate = performSubstrateRegresssion(args['psr'],sub_plate,args,directory)

        # format name and save
        df = sub_plate.key
        df_path = assemblePath(directory['summary'],pid,'.txt')       
        df.to_csv(df_path,sep='\t',header=True,index=True)

        #endfor

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


def printDirectoryContents(directory,sort=True,tab=True):
    '''
    prints a list of contents in a directory

    Args:
        directory (str): path
        sort (boolean)
        tab (boolean): pads each item with (4 spaces) in front

    Returns:
        items (list) where elements are file names (str)
    '''

    items = os.listdir(directory)
    
    if sort:
        items = sorted(items)

    if tab:
        items = ['    {}'.format(i) for i in items]

    return items


def tidyDictPrint(input_dict):
    '''
    Returns a message that neatly prints a dictionary into multiple lines. Each line
        is a key:value pair. Keys of dictionary are padded with period. Padding is 
        dynamically selected based on longest argument.
    '''

    # dynamically set width of padding based on maximum argument length
    args = input_dict.keys()
    max_len = float(len(max(args,key=len))) # length of longest argument
    width = int(np.ceil(max_len/10)*10) # round up length to nearest ten

    # if width dos not add more than three padding spaces to argument, add 5 characters
    if (width - max_len) < 4:
        width += 5

    # compose multi-line message
    msg = ''
    for arg,value in input_dict.items():
        msg += '{:.<{width}}{}\n'.format(arg,value,width=width)

    return msg


def tidyMessage(msg):
    '''
    Returns a message in a text-based banner with header and footer
        composed of dashes and hashtags. Messae will be paddedd with flanking
        spaces. Length of message determines length of banner. Heigh of banner is 4 lines.

    Args:
        msg (str)

    Returns:
        msg_print (str)
    '''

    msg_len = len(msg)+2
    banner = '#{}#'.format('-'*msg_len)
    msg_print = '{}\n# {} #\n{}\n'.format(banner,msg,banner)

    return msg_print


def validateDirectories(directory,verbose=False):
    '''
    Checks specific directories for their existence and makes sure data was 
        provided by user. It will also compose and can print a message that can
        communicate with user the results of this validation.

    Args:
        directory (dict): dictionary where keys are AMiGA-relevant folders and 
            values are the corresponding file paths
        verbose (boolean): if True print message
 
    Returns:
        None
    '''

    # params (dict) defines parameters for directory validation
    #     
    #     key: directory in directory-storing dictionary (str) 
    #
    #     value (4-tuples) where:    
    #     1. generic name of directory for communcating with user (str)
    #     2. verbosity: whether to communicate with user or not (boolean)
    #     3. initialization: whether to initialize directory or not (boolean)
    #     4. force system exit: whether to exist system if error detected

    params = {
        'parent':('Input directory',False,True),
        'data':('Data directory',False,False),
        'derived':('Derived data directory',True,False),
        'mapping':('Mapping directory',True,False),
        'summary':('Summary directory',True,False),
        'figures':('Figures directory',True,False)
    }

    full_msg = ''

    max_width = 25 # maximum generic name length is 22, add 3 for padding

    # check whether these directories exist and/or take action based on parameters 
    for i,params_i in params.items():
        _,msg = checkDirectoryExists(directory[i],*params_i)
        full_msg += msg

    # check whether user provided at least one data file
    _,msg = checkDirectoryNotEmpty(directory['data'],params['data'][0])
 
    full_msg += '\n' # message will multi-line and bottom is padded with a new line
    full_msg += msg # itemized list of content of data folder

    if verbose:
        print(full_msg)

    return None


