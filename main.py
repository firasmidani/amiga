#!/usr/bin/env python

'''
DESCRIPTION main driver script for Analysis of Microbial Growth Assays (AMiGA)
'''

__author__ = "Firas Said Midani"
__version__ = "0.1.0"
__email__ = "midani@bcm.edu"

from libs import aio
from libs.config import config

#----------------------#
# Parsing user command #
#----------------------#

# parse terminal command for arguments
args = aio.parseCommand(config);

# did the user provide a path that points to a file or directory?
parent,filename = aio.isFileOrFolder(args['fpath']) 

# communicate with user
print(aio.tidyMessage('AMiGA is peeking inside the working directory'))

# define file paths for AMiGA-relevant folders (dict)
directory = aio.mapDirectories(parent)

# define file paths for AMiGA-relevant files (dict)
files = aio.mapFiles(directory)

# validate working directory structure and contents (no output)
aio.validateDirectories(directory,verbose=args['verbose'])

# communicate with user
print(aio.tidyMessage('AMiGA is parsing command-line arguments and parameter files'))

# interpret terminal command or text files for parameters (dict)
params = aio.interpretParameters(files,args,verbose=args['verbose'])

# parse meta.txt file if it exists (df,list)
df_meta,df_meta_plates = aio.checkMetaText(files['meta'],verbose=args['verbose'])

# communicate with user
print(aio.tidyMessage('AMiGA is parsing and cleaning data files'))

# parse data files (dict)
data = aio.readPlateReaderFolder(filename,directory,config,save=args['sdd'],verbose=args['verbose'])

# communicate with user
print(aio.tidyMessage('AMiGA is parsing and reading mapping files'))

# parse mapping files and mapping data (dict)
mappings = aio.assembleMappings(data,directory['mapping'],files['meta'],verbose=args['verbose'])

# communicate with user
print(aio.tidyMessage('AMiGA is trimming mapping files based on user input'))

# trim mapping data based on user input
mappings = aio.trimMappings(mappings,params,verbose=args['verbose'])

