#!/usr/bin/env python

'''
DESCRIPTION library for input/output commands for AMiGA
'''

__author__ = "Firas Said Midani"
__vesrion__ = "0.1.0"
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

from codecs import open as codecs_open
from codecs import BOM_UTF8,BOM_UTF16_BE,BOM_UTF16_LE,BOM_UTF32_BE,BOM_UTF32_LE

# define variables
interval_default = 600


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

    sep = ['' if copydirectory[-1]=='/' else '/'][0]
    newfilepath = '{}{}{}.tsv'.format(copydirectory,sep,filebase)

    return filename, filebase, newfilepath
    

def parseCommand():
    '''
    Interprets the arguments passed by the user to AMiGA. If the verobse argument 
        is set to True, argumentParser will also print a message summarizing the 
        the user-passed command-line arguments.

    Note: Function is AMiGA-specific and should not be used verbatim for other apps.

    Returns:
        args (dict): a dictionary with keys as suggested variable names
            and keys as the user-passed and argparse-interpreted arguments.
    '''

    args_dict= {};

    # parse arguments 
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--input',required=True)
    parser.add_argument('-f','--flag',required=False)
    parser.add_argument('-s','--subset',required=False)
    parser.add_argument('-p','--hypothesis',required=False)
    parser.add_argument('-t','--interval',required=False)
    parser.add_argument('-v','--verbose',action='store_true',default=False)
    parser.add_argument('--only-plot-plate',action='store_true',default=False)

    # pass arguments to local variables 
    args = parser.parse_args()
    args_dict['fpath'] = args.input  # File path provided by user
    args_dict['flag'] = args.flag
    args_dict['subset'] = args.subset
    args_dict['hypothesis'] = args.hypothesis
    args_dict['interval'] = args.interval
    args_dict['verbose'] = args.verbose
    args_dict['only_plot_plate'] = args.only_plot_plate

    # summarize command-line artguments and print
    if args_dict['verbose']:
        msg = '\n'
        msg += tidyMessage('User provided the following command-line arguments:')
        msg += '\n' 
        msg += tidyDictPrint(args_dict)
        print(msg)

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
        filepath (str): path to meta.txt
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
        print(df_meta.Plate_ID)
    except:
        df_meta_plates = []

    # neatly prints meta.txt to terminal
    if verbose and exists:
        tab = tabulate.tabulate(df_meta,headers='keys',tablefmt='psql')
        msg = '{:.<16}{}\n{}\n'.format('Meta-Data is',filepath,tab)
        print(msg)
    elif verbose:
        print('No meta.txt file found\n')

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
        directory[child] = '{}/{}'.format(directory['parent'],child)

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
    children = ['flag','hypothesis','subset','interval']

    for child in children:
        files[child] = '{}/{}.txt'.format(directory['parameters'],child)

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
        ('hypothesis','\+|,',False)
    ]
    
    # initialize all parameters based on their settings
    params_dict = {}
    for pp,sep,integerize in params_settings:
        params_dict[pp] = initializeParameter(files[pp],args[pp],sep=sep,integerize=integerize)

    if verbose:
        print(tidyDictPrint(params_dict))

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


#def parseData(filename,args,directory,parms,interval=default_interval):
#    '''
#    '''
#
#    data = {}
#    if filename:
#        list_data = [filename]
#        filepath = args['fpath']
#        filebase = ''.join(filename.split('.')[:-1]) # name without extenstion (i.e. Plate_ID)
#        interval = [params['interval'][0] if (filebase in interval_dict.keys()) else default_interval][0]
#        data[filebase] = plates.readPlateReaderData(
#            filepath,interval=interval,save=True,save_dirname=directory['derived'],
#            interval=default_interval,interval_dict=interval_dict)
#    else:
#        list_data = sorted(os.listdir(directory['data']))
#        data = readPlateReaderFolder(
#            folderpath=directory['data'],save=True,save_dirname=directory['derived'],
#            interval=default_interval,interval_dict 

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
            sep = ['' if dirpath[-1]=='/' else '/'][0]
            list_files.append('{}{}{}'.format(dirpath,sep,filename))
  
    return list_files


def findFirstRow(filepath,encoding):
    '''
    Searches for line beginning with the Well ID "A1", determines the number of
        rows that need to be skipped for reading this line, and indicates if
        index column was not found.

    Args:
        filepath (str)

    Returns:
        count (int): number of lines that need to be skipped to read data for first well
        index_column (0 or None): location of row namess, if found as first character (0) or not found (None) 
    '''

    fid = open(filepath,'r',encoding=encoding)

    count = 0
    for line in fid.readlines():
        if line.startswith('A1'):
            fid.close()
            index_column = 0  # row names are the zero-indexed column
            return count, index_column
        count += 1
    else:
        fid.close()
        count = 0
        index_column = None  # row names were not found
        return count, index_column


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
 

def readPlateReaderData(filepath,interval,copydirectory):
    '''
    Reads a single file and adjusts it to AMiGA-desired format. The desired format is 
        time point (row) by well (col, where first column is actual time point, so row
        names are arbitray numerical order (1...n). All cells should be floats.

    Args:
        filepath (str)
        interval (int): time interval length
        copydirectory (str): location  
    '''

    filename,filebase,newfilepath = breakDownFilePath(filepath,copydirectory)

    # make sure file is ASCII- or BOM-encoded
    #filepath = checkFileEncoding(filepath)
    encoding = checkFileEncoding(filepath)


    # identify number of rows to skip and presence/location of index column (i.e. row names)
    skiprows,index_col = findFirstRow(filepath,encoding=encoding)

    # read tab-delimited data file
    df = pd.read_csv(filepath,sep='\t',header=None,index_col=index_col,skiprows=skiprows)

    # explicitly define time series based on data size and time interval 
    df.columns = listTimePoints(interval=interval,numTimePoints=df.shape[1])

    # if index column is absent, create one 
    if index_col == None:
        df.index = parseWellLayout(order_axis=0).index.values

    # explicilty assign column names 
    df.index.name = 'Well' # well disappears
    df.T.index.name = 'Time'

    # remove columns (time points) with only NA values (sometimes happends in plate reader files)
    df = df.iloc[:,np.where(~df.isna().all(0))[0]]
    df = df.astype(float)

    # set to following format: time point (row) by well (column)
    df = df.T.reset_index(drop=False) # values are OD (float) except first column is time (float)

    # save derived data copy in proper location
    df.to_csv(newfilepath,sep='\t',header=True)  # does it save header index name (i.e. Time)

    return df

def parseWellLayout():
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
    rows = list(string.ascii.uppercase[0:8])
    cols = list(str(ii) for ii in range(1,13))

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


def readPlateReaderFolder(filename,directory,save=False,interval_dict={},verbose=False):
    '''
    Finds, reads, and formats all files in a directory to be AMiGA-compatible. 
    '''

    folderpath = directory['data']
    copydirectory = directory['derived']

    # user may have passed a specific file or a directory to the input argument
    if filename:
        filepaths = '{}/{}'.format(folderpath,filename)
    else:
        filepaths = findPlateReaderFiles(folderpath)

    # read one data file at a time
    df_dict = {}
    for filepath in sorted(filepaths):

        print('Reading {}'.format(filepath))

        # get extension-free file name and path for derived copy
        _, filebase, newfilepath = breakDownFilePath(filepath,copydirectory=copydirectory)

        # set the interval time
        if filebase in interval_dict.keys():
            plate_interval = interval_dict[filebase][0]
        else:
            plate_interval = interval_default

        # read and adjust file to format: time by wells where first column is time and rest are ODs
        df = readPlateReaderData(filepath,plate_interval,copydirectory)
        df_dict[filebase] = df

    return df_dict


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

    # if width dos not add any padding to argument, add 5 characters
    if width == max_len:
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
(
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
    #     3. initialization: whetehr to initialize directory or not (boolean)
    #     4. force system exit: whetehr to exist system if error detected

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


