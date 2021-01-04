#!/usr/bin/env python

'''
AMiGA wrapper: main driver script for Analysis of Microbial Growth Assays (AMiGA).
'''

__author__ = "Firas S Midani"
__email__ = "midani@bcm.edu"


from libs.analyze import basicSummaryOnly, runGrowthFitting, runCombinedGrowthFitting
from libs.comm import tidyMessage
from libs.config import config
from libs.detail import assembleMappings
from libs.interface import parseCommand, interpretParameters
from libs.org import isFileOrFolder, mapDirectories, mapFiles, validateDirectories
from libs.read import readPlateReaderFolder
from libs.test import HypothesisTest
from libs.trim import trimInput


# parse terminal command for arguments
args = parseCommand(config);

# did the user provide a path that points to a file or directory?
parent,filename = isFileOrFolder(args['fpath']) 

# communicate with user
print()
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
params,args = interpretParameters(files,args,verbose=args['verbose'])

# communicate with user
print(tidyMessage('AMiGA is parsing and cleaning data files'))

# parse data files (dict)
data = readPlateReaderFolder(filename,directory,interval=params['interval'],save=args['scd'],verbose=args['verbose'])

# communicate with user
print(tidyMessage('AMiGA is parsing and reading mapping files'))

# parse mapping files and mapping data (dict)
mappings = assembleMappings(data,directory['mapping'],files['meta'],save=args['smt'],verbose=args['verbose'])

# plot and summarize plate(s), if it is the only request
basicSummaryOnly(data,mappings,directory,args,verbose=args['verbose'])

# communicate with user
print(tidyMessage('AMiGA is preparing or analyzing data based on user input'))

# run hypothesis testing, if requested
HypothesisTest(mappings,data,params,args,directory)

# trim mapping data based on user input
data,mappings = trimInput(data,mappings,params,nskip=args['nskip'],verbose=args['verbose'])

# communicate with user
print(tidyMessage('AMiGA is fitting growth curves'))

# run growth fitting 
runGrowthFitting(data,mappings,directory,args,verbose=args['verbose'])

