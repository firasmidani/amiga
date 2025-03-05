#!/usr/bin/env python

'''
AMiGA library of functions for reading and cleaning data.
'''

__author__ = "Firas S Midani"
__email__ = "midani@bcm.edu"


# TABLE OF CONTENTS (8 functions)

# readPlateReaderFolder
# findPlateReaderFiles
# breakDownFilePath
# readPlateReaderData
# checkFileEncoding
# findRowsAndIndex
# isWellId
# listTimePoints

import os
import sys
import numpy as np # type: ignore
import pandas as pd # type: ignore

from string import ascii_uppercase

from .config import config
from .detail import parseWellLayout
from .comm import smartPrint
from .org import assemblePath


def readPlateReaderFolder(filename=None,directory=None,interval=dict(),save=False,verbose=False):
    '''
    Finds, reads, and formats all files in a directory to be AMiGA-compatible.

    Args:
        filename (str or None): 
            if str: path to a single data file to be read.
            if None: user is interested in reading one or more data files (so user must pass directory argument).
        directory (dictionary or str or None):
            if dictionary: Keys are folder names, values are their paths. Keys must include 'data' and 'derived'
                'data' sub-folder must exist and house one or more data files to be read. 
            if str: path to folder that houses one ore more data files to be read.
            if None: user is interested in reading a single data file (so user must pass filename argument).
        interval (dictionary or numeric):
            if numeric: must be int or float.
            if dictionary: Keys are file names, values are their respective interval parameter, e.g. 
                {'CD2015_PM1-1':600,'CD2048_PM1-1':900}). If a filename does not have a corresponding key in the
                dictionary, the default parameter for 'interval' in the 'config' dictionary will be used. 
        save (boolean): will save AMiGA-formatted file in the 'derived' or input folder as a TSV file.
        verbose (boolean)
    '''

    if (filename is None) and (directory is None): 
        sys.exit('FATAL USER ERROR: User must pass either a filename or a directory argument')

    # what is the data folder (folderpath) and where to save formatted data (copydirectory)? 
    if isinstance(directory,dict):
        folderpath = directory['data']
        copydirectory = directory['derived']
    elif isinstance(directory,str):
        copydirectory = folderpath = directory
    elif directory is None:
        copydirectory = folderpath = os.path.dirname(filename)

    # user may have passed a specific file or a directory to the input argument
    if filename:
        filepaths = [f'{folderpath}{os.sep}{filename}']
    else:
        filepaths = findPlateReaderFiles(folderpath)
    # either way, filepaths must be an iterable list or array

    # read one data file at a time
    df_dict = {}
    for filepath in sorted(filepaths):

        # communicate with user
        smartPrint(f'Reading {filepath}',verbose)

        # get extension-free file name and path for derived copy
        _, filebase, newfilepath = breakDownFilePath(filepath,copydirectory=copydirectory)

        # set the interval time
        if isinstance(interval,(int,float)):
            plate_interval = float(interval)
        elif filebase in interval.keys():
            plate_interval = interval[filebase]
        else: 
            plate_interval = config['interval']

        # read and adjust file to format: time by wells where first column is time and rest are ODs
        df = readPlateReaderData(filepath,plate_interval,copydirectory,save=save)
        df_dict[filebase] = df#..iloc[nskip:,:]

    smartPrint('',verbose)  # print empty newline, for visual asethetics only

    return df_dict


def findPlateReaderFiles(directory,extension=None):
    '''
    Recrusivelys searches a directory for all files with specific extensions.

    Args:
        directory (str): path to data directory

    Returns:
        list_files (list): list of of paths to data files
    '''

    # you can modify this to include other extensions, but internal data must still be tab-delimited 
    if extension is None:
        extension = ('.txt','TXT','tsv','TSV','asc','ASC')

    # recursively walk through a directory, if nested
    list_files = []

    for (dirpath,dirnames,filenames) in os.walk(directory):

        # only keep files with acceptable extensions
        filenames = [ii for ii in filenames if ii.endswith(extension)]

        # compose and store filepaths but avoid double slashes (i.e. //) between directory names
        for filename in filenames:

            list_files.append(assemblePath(dirpath,filename))
  
    return list_files


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
    filebase = '.'.join(filename.split('.')[:-1])

    newfilepath = assemblePath(copydirectory,filebase,'.tsv')

    return filename, filebase, newfilepath
    

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
    skiprows,skipfooter,index_col = findRowsAndIndex(filepath,encoding=encoding)

    # read tab-delimited data file
    df = pd.read_csv(filepath,sep='\t',header=None,index_col=index_col,skiprows=skiprows,skipfooter=skipfooter,encoding=encoding,engine='python')

    # explicitly define time series based on data size and time interval 
    df.columns = listTimePoints(interval=interval,numTimePoints=df.shape[1])

    # if index column is absent, create one 
    if index_col is None:
        nrows = df.shape[0]
        df.index = parseWellLayout(order_axis=0).index[0:nrows].values

    # explicilty assign column names 
    df.T.index.name = 'Time'

    # remove columns (time points) with only NA values (sometimes happens in plate reader files)
    df = df.iloc[:,np.where(df.notna().all(axis=0))[0]]

    # remove rows (smples) with only NA values (happens if there is meta-data in file after measurements)
    df = df.iloc[np.where(df.T.notna().all(axis=0))[0],:]

    # strip leading zeros in row or well IDs
    df.index = [ii[0] + ii[1:].lstrip('0') for ii in df.index]

    try:
        df = df.astype(float)
    except Exception:
        msg = f'\nFATAL DATA ERROR: {filename} data file is not properly formatted. '
        msg += 'AMiGA expected measurement values that are numerical integers or floats '
        msg += 'but instead detected strings (text characters). '
        msg += 'Please see documentation for instructions on how to properly format data files. \n'
        sys.exit(msg)

    # set to following format: time point (row) by well (column)
    df = df.T.reset_index(drop=False) # values are OD (float) except first column is time (float)

    # save derived data copy in proper location
    if save:
        df.to_csv(newfilepath,sep='\t',header=True)  # does it save header index name (i.e. Time)

    return df


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
            open(filepath,encoding=encoding).readline()
            return encoding
        except UnicodeDecodeError:
            pass

    # exit 
    msg = f'FATAL DATA ERROR: AMiGA cannot read{filepath}. '
    msg += 'AMiGA can only read data text files encoded with either '
    msg += 'UTF-8, UTF-16, UTF-32, or ASCII.\n'
    sys.exit(msg) 
 

def findRowsAndIndex(filepath,encoding):
    '''
    Searches for line that begins with a Well ID (defined as a letter followed by digits), 
        determines the number of rows that need to be skipped for reading this line, and
        indicates if index column was not found. Then, reverse reads file and searches for
        line that begins with a Well ID to detemine the number of lines to skip in footer. 

    Args:
        filepath (str)

    Returns:
        firstrow (int): number of lines that need to be skipped to read data for first well
        lasttrow (int): number of lines at the bottom that need to be skipped to read data to step at last ro
        index_column (0 or None): location of row namess, if found as first character (0) or not found (None) 
    '''

    fid = open(filepath,encoding=encoding)

    firstrow = 0
    for line in fid.readlines():
        line_start = line.strip().split('\t')[0]
        if isWellId(line_start):
            index_column = 0  # row names are the zero-indexed column
            break
        firstrow += 1
    else:
        firstrow = 0
        lastrow = 0
        index_column = None  # row names were not found

    fid.close()

    # I could have tweaked the above for-else statement to detect the footer using the same iterator, BUT
    #     that would have required reading eahc line of the file. If the file is very large, this can 
    #     increase time. Instead, I simply reverse the iterated lines and stop reading once Well ID is detected.

    fid = open(filepath,encoding=encoding)

    lastrow = 0
    for line in reversed(fid.readlines()):
        line_start = line.strip().split('\t')[0]
        if isWellId(line_start):
            break
        lastrow += 1

    fid.close()

    return firstrow,lastrow,index_column


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

    if (item.lstrip('0')[0] in ascii_uppercase) and (item[1:].isdigit()):
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

