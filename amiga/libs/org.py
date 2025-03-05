#!/usr/bin/env python

'''
AMiGA library of functions for parsing and organizing the working directory.
'''

__author__ = "Firas S Midani"
__email__ = "midani@bcm.edu"


# TABLE OF CONTENTS (9 functions)

# validateDirectories
# checkDirectoryExists
# checkDirectoryNotEmpty
# printDirectoryContents
# mapDirectories
# mapFiles
# isFileOrFolder
# assemblePath
# assembleFileName

import os
import sys


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
    #     2. initialization: whether to initialize directory or not (boolean)
    #     3. force system exit: whether to exist system if error detected

    params = {
        'parent':('Input directory',False,True),
        'data':('Data directory',False,False),
        'derived':('Derived data directory',True,False),
        'mapping':('Mapping directory',True,False),
        'summary':('Summary directory',True,False),
        'figures':('Figures directory',True,False),
        'models':('Models directory',True,False)
    }

    full_msg = ''

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
        sys.exit(f'FATAL USER ERROR: {generic_name} {directory} does not exist.\n')
    elif initialize:
        os.makedirs(directory)
        arg = True
        msg = f'WARNING: {directory} did not exist but was created.\n'
    else:
        arg = False
        msg = f'WARNING: {directory} does not exist.\n'

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
        sys.exit(f'FATAL USER ERROR: {generic_name} {directory} is empty.\n')
    else:
        arg = True
        msg = f'{generic_name} {directory} has {numFiles} files:'
        msg += '\n\n'
        msg += '\n'.join(printDirectoryContents(directory)) # print one item per line
        msg += '\n' # pad bottom withe new line

    return arg,msg


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
        items = [f'    {i}' for i in items]

    return items


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
    children = ['data','derived','mapping','summary','parameters','figures','models']

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
    files['meta'] = '{}{}{}.txt'.format(directory['mapping'],os.sep,'meta')

    # format paths for files in the 'parameter' sub-directory
    children = ['flag','hypothesis','subset','interval']

    for child in children:

        files[child] = assemblePath(directory['parameters'],child,'.txt')

    return files


def isFileOrFolder(filepath,up=2):
    '''
    Determines if a path points to a file or directory.

    Args:
        filepath (str): file path 
        up (int): which directoy to return if a file, the parent (1) or grandparent (2)
    '''

    isFile = os.path.isfile(filepath)

    if isFile:
        if up == 2:
            parent = os.path.dirname(os.path.dirname(filepath))
        elif up == 1:
            parent = os.path.dirname(filepath)
        filename = os.path.basename(filepath)
        if parent == '' or parent is None:
            parent = '.'
        return parent,filename
    else:
        parent = filepath
        return parent,None


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
    sep = ['' if directory[-1]==os.sep else os.sep][0]
    file_path = f'{directory}{sep}{filebase}{extension}'

    return file_path


def assembleFullName(folder,prefix,filename,suffix,extension):
    '''
    Assembles a file name using the arguments into a complete full path. 

    Args:
        folder (str): full path to file
        prefix (str): prefix to filename (precedes underscore)
        filename (str): filename without extension
        suffix (str): suffix to filename (succedes underscore)
        extension (str): file extension (e.g. .txt,.tsv,.pdf). should include period. 

    Returns:
        file_path (str): full path to generated file name
    '''

    file_name = f'{prefix}_{filename}_{suffix}'
    file_name = file_name.strip('_') # in case prefix is empty

    file_path = assemblePath(folder,file_name,extension)

    return file_path

