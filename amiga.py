#!/usr/bin/env python

'''
Main driver script for the Analysis of Microbial Growth Assays (AMiGA).
'''

__author__ = "Firas Said Midani"
__version__ = "0.2.0"
__email__ = "midani@bcm.edu"


from libs.config import config
from libs.analyze import basicSummaryOnly,runGrowthFitting
from libs.test import testHypothesis
from libs.detail import assembleMappings
from libs.org import isFileOrFolder, mapDirectories, mapFiles, validateDirectories
from libs.read import readPlateReaderFolder
from libs.interface import parseCommand, interpretParameters
from libs.comm import tidyMessage
from libs.trim import trimInput


# parse terminal command for arguments
args = parseCommand(config);

# did the user provide a path that points to a file or directory?
parent,filename = isFileOrFolder(args['fpath']) 

# communicate with user
print(tidyMessage('AMiGA is peeking inside the working directory'))

# define file paths for AMiGA-relevant folders (dict)
directory = mapDirectories(parent)

# define file paths for AMiGA-relevant files (dict)
files = mapFiles(directory)

# validate working directory structure and contents (no output)
validateDirectories(directory,verbose=args['verbose'])

# communicate with user
print(tidyMessage('AMiGA is parsing command-line arguments and parameter files'))

# interpret terminal command or text files for parameters (dict)
params = interpretParameters(files,args,verbose=args['verbose'])

# communicate with user
print(tidyMessage('AMiGA is parsing and cleaning data files'))

# parse data files (dict)
data = readPlateReaderFolder(filename,directory,config,save=args['sdd'],verbose=args['verbose'])

# communicate with user
print(tidyMessage('AMiGA is parsing and reading mapping files'))

# parse mapping files and mapping data (dict)
mappings = assembleMappings(data,directory['mapping'],files['meta'],verbose=args['verbose'])

# plot plate(s), if it is the only request
basicSummaryOnly(data,mappings,directory,args,verbose=args['verbose'])

# communicate with user
print(tidyMessage('AMiGA is preparing data based on user input'))

# run hypothesis testing, if requested
testHypothesis(data,mappings,params,args,directory,subtract_control=args['sc'],sys_exit=True,verbose=args['verbose'])

# trim mapping data based on user input
data,mappings = trimInput(data,mappings,params,verbose=args['verbose'])

# communicate with user
print(tidyMessage('AMiGA is fitting growth curves'))

# run growth fitting 
runGrowthFitting(data,mappings,directory,args,config,verbose=args['verbose'])

