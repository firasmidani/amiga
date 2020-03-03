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

# annotate mapping files based on user input (dict)
#mappings = aio.annotateMappings(mappings,params,verbose=args['verbose'])

# plot plate(s), if it is the only request
aio.plotPlatesOnly(data,mappings,directory,args,verbose=args['verbose'])

# communicate with user
print(aio.tidyMessage('AMiGA is trimming samples based on user input'))

# trim mapping data based on user input
data,mappings = aio.trimInput(data,mappings,params,verbose=args['verbose'])

# communicate with user
print(aio.tidyMessage('AMiGA is preparing data for growth curve fitting'))

# run hypothesis testing, if requested
aio.testHypothesis(data,mappings,params,verbose=args['verbose'])

# run growth fitting 
aio.runGrowthFitting(data,mappings,verbose=args['verbose'])

