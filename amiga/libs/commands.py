#!/usr/bin/env python

'''
AMiGA wrapper: main driver script for Analysis of Microbial Growth Assays (AMiGA).
'''

__author__ = "Firas S Midani"
__email__ = "midani@bcm.edu"


# TABLE OF CONTENTS (1 class with 4 class functions)

# Command
# __init__
# summarize
# test
# fit

from .analyze import basicSummaryOnly, runGrowthFitting
from .comm import tidyMessage
from .config import config
from .detail import assembleMappings
from .interface import interpretParameters
from .org import isFileOrFolder, mapDirectories, mapFiles, validateDirectories
from .read import readPlateReaderFolder
from .test import HypothesisTest
from .trim import trimInput, flagWells


class Command:

    def __init__(self,args,config=config):

        # parse terminal command for arguments
        #args = parseCommand(args,config);

        # did the user provide a path that points to a file or directory?
        parent,filename = isFileOrFolder(args.input) 

        # communicate with user
        print()
        print(tidyMessage('AMiGA is peeking inside the working directory'))

        # define file paths for AMiGA-relevant folders (dict)
        directory = mapDirectories(parent)

        # define file paths for AMiGA-relevant files (dict)
        files = mapFiles(directory)

        # validate working directory structure and contents (no output)
        validateDirectories(directory,verbose=args.verbose)

        # communicate with user
        print(tidyMessage('AMiGA is parsing command-line arguments and parameter files'))

        # interpret terminal command or text files for parameters (dict)
        params,args = interpretParameters(files,args,verbose=args.verbose)

        # communicate with user
        print(tidyMessage('AMiGA is parsing and cleaning data files'))

        # parse data files (dict)
        data = readPlateReaderFolder(filename,directory,interval=params['interval'],
            save=args.save_cleaned_data,verbose=args.verbose)

        # communicate with user
        print(tidyMessage('AMiGA is parsing and reading mapping files'))

        # parse mapping files and mapping data (dict)
        mappings = assembleMappings(data,directory['mapping'],files['meta'],
            save=args.save_mapping_tables,verbose=args.verbose)

        self.directory = directory
        self.params = params
        self.args = args
        self.data = data
        self.mappings = mappings

    def summarize(self):

        # flag wells
        self.mappings = flagWells(self.mappings,self.params['flag'],verbose=self.args.verbose,drop=False)

        # communicate with user
        print(tidyMessage('AMiGA is summarizing and plotting data'))

        # plot and summarize plate(s), if it is the only request
        basicSummaryOnly(self.data,self.mappings,self.directory,self.args,verbose=self.args.verbose)


    def test(self):

        # communicate with user
        print(tidyMessage('AMiGA is testing data based on user input'))

        # run hypothesis testing, if requested
        HypothesisTest(self.mappings,self.data,self.params,self.args,self.directory)


    def fit(self):

        args = self.args

        # communicate with user
        print(tidyMessage('AMiGA is preparing data based on user input'))

        # trim mapping data based on user input
        data,mappings = trimInput(self.data,self.mappings,self.params,nskip=args.skip_first_n,verbose=args.verbose)

        # communicate with user
        print(tidyMessage('AMiGA is fitting growth curves'))

        # run growth fitting 
        runGrowthFitting(data,mappings,self.directory,args,verbose=args.verbose)

