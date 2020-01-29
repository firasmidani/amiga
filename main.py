#!/usr/bin/env python

'''
DESCRIPTION main driver script for Analysis of Microbial Growth Assays (AMiGA)
'''

__author__ = "Firas Said Midani"
__verion__ = "0.1.0"
__email__ = "midani@bcm.edu"

from libs import io

#----------------------#
# Parsing user command #
#----------------------#

# initialize variables
directory = {}
mapping = {}
files = {}
data = {}

# parse terminal command for arguments
args = io.parseCommand();

# did the user provide a path that points to a file or directory?
parent,filename = libs.io.isFileOrFolder(args['fpath']) 

# communicate with user
print(io.tidyMessage('AMiGA is peeking inside the working directory'))

# define file paths for AMiGA-relevant folders (dict)
directory = io.mapDirectories(parent)

# define file paths for AMiGA-relevant files (dict)
files = io.mapFiles(directory)

# validate working directory structure and contents
io.validateDirectories(directory,verbose=args['verbose'])

# communicate with user
print(io.tidyMessage('AMiGA is parsing command-line arguments and parameter files'))

# interpret terminal command or text files for parameters
io.interpretParameters(files,args,verbose=args['verbose'])

# parse meta.txt file if it exists
df_meta,df_meta_plates = io.checkMetaText(files['meta'],verbose=args['verbose'])

# communicate with user
print(io.tidyMessage('AMiGA is parsing and cleaning data files'))

# parse data files
io.readPlateReaderFolder(filename,directory,save=True,verbose=args['verbose'])

# communicate with user
print(io.tidyMessage('AMiGA is parsing and reading mapping files'))

# parse mapping files

