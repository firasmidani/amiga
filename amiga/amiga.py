#!/usr/bin/env python

'''
AMiGA library for the AMiGA class for parsing command-line arguments. 
'''

__author__ = "Firas S Midani"
__email__ = "midani@bcm.edu"


# TABLE OF CONTENTS (1 class with 8 sub-functions and 1 auxiliary function)

# AMiGA (CLASS)
#   __init__
#   compare
#   fit
#   heatmap
#   normalize
#   print_defaults
#   summarize
#   test
#
# print_arguments

import os
import sys
import argparse
import importlib.metadata


# Add the parent directory to sys.path to allow package imports
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from amiga.libs.commands import Command
from amiga.libs.comm import tidyMessage, tidyDictPrint
from amiga.libs.config import config

# define long string variables

usage_init = '''amiga <command> [<args>]

The most commnly used amiga commands are:
    summarize       Perform basic summary and plot curves
    fit             Fit growth curves
    normalize       Normalize growth parameters of fitted curves
    compare         Compare summary statistics for two growth curves
    test            Test a specific hypothesis
    heatmap         Plot a heatmap
    get_confidence  Compute confidence intervals for parameters or curves
    get_time        Get time at which growth reaches a certain value
    print_defaults  Shows the default values stored in libs/config.py

See `amiga <command> --help` for information on a specific command.
For full documentation, see https://firasmidani.github.io/amiga
'''

# define auxiliary functions

def print_arguments(args):

    msg = '\n'
    msg += tidyMessage('User provided the following command-line arguments:')
    msg += '\n'
    msg += tidyDictPrint(vars(args))

    print(msg)


class AMiGA:
    '''
    Class for interpreting the arguments passed by the user to AMiGA. If the verobse argument 
        is set to True, sub-commands will also print a message summarizing the 
        the user-passed command-line arguments.
    '''

    def __init__(self):
        parser = argparse.ArgumentParser(
            usage=usage_init)
        parser.add_argument('command', help='Subcommand to run. See amiga --help for more details.')
        parser.add_argument('-v','--version', action="version", version=f"amiga {importlib.metadata.version('amiga')}", help="Show version and exit.")

        args = parser.parse_args(sys.argv[1:2])

        if not hasattr(self,args.command):
            print(f"Error: Unknown command '{args.command}'\n")
            parser.print_help()
            exit(1)
        else: 
            getattr(self, args.command)()

        # Prevent Python from returning the object representation
        sys.exit(0)

    def compare(self):

        parser = argparse.ArgumentParser(
            description='Compare two growth curves')

        parser.add_argument('-i','--input',required=True,action='append')
        parser.add_argument('-o','--output',required=True,help='ouptut filename including path')
        parser.add_argument('-s','--subset',required=True)
        parser.add_argument('--confidence',required=False,type=float,default=95,
            help='Must be between 80 and 100. Default is 95.')
        parser.add_argument('--verbose',action='store_true',default=False)

        if len(sys.argv) == 2:
            parser.print_help()
            sys.exit(0)

        args = parser.parse_args(sys.argv[2:])

        if args.verbose:
            print_arguments(args)

        if (args.confidence < 80) | (args.confidence > 100):
            msg = 'FATAL USER ERROR: Confdience must be between 80 and 100.'
            sys.exit(msg)
        
        from amiga.libs.compare import main as compare_main

        compare_main(args)


    def get_confidence(self):

        parser = argparse.ArgumentParser(
            description = 'Compute confidence intervals for parameters or curves.')
        parser.add_argument('-i','--input',required=True)
        parser.add_argument('--type',required=True,default='Parameters',choices=['Parameters','Curves'])
        parser.add_argument('--confidence',required=False,type=float,default=95,
            help='Must be between 80 and 100. Default is 95.')
        parser.add_argument('--include-noise',action='store_true',default=False,
            help='Include the estimated measurement noise when computing confidence interval (For Curves Only).')
        parser.add_argument('--over-write',action='store_true',default=False,
            help='Over-write file otherwise a new copy is made with "_confidence" suffix')
        parser.add_argument('--verbose',action='store_true',default=False)

        if len(sys.argv) == 2:
            parser.print_help()
            sys.exit(0)

        args = parser.parse_args(sys.argv[2:])

        if args.verbose:
            print_arguments(args)

        if (args.confidence < 80) | (args.confidence > 100):
            msg = 'FATAL USER ERROR: Confdience must be between 80 and 100.'
            sys.exit(msg)
        
        from amiga.libs.confidence import main as get_confidence_main
        
        get_confidence_main(args)


    def get_time(self):

        parser = argparse.ArgumentParser(
            description='Get time at which OD reaches a certain value')

        parser.add_argument('--gp-data',required=True)
        parser.add_argument('--summary',required=True)
        parser.add_argument('--threshold',required=True,type=float)
        parser.add_argument('--curve-format',required=False,default='OD_Growth_Fit',
            choices=['OD_Data','OD_Fit','GP_Input','GP_Output',
            'OD_Growth_Fit','OD_Growth_Data','GP_Derivative'])

        if len(sys.argv) == 2:
            parser.print_help()
            sys.exit(0)

        args = parser.parse_args(sys.argv[2:])

        from amiga.libs.thresholds import get_time_main

        get_time_main(args)


    def fit(self):

        parser = argparse.ArgumentParser(
            description='Fit growth curves')

        # defining input/output
        parser.add_argument('-i','--input',required=True)
        parser.add_argument('-o','--output',required=False)
        parser.add_argument('-f','--flag',required=False)
        parser.add_argument('-s','--subset',required=False)
        parser.add_argument('-t','--interval',required=False)
        parser.add_argument('-tss','--time-step-size',action='store',type=int,default=1)#11
        parser.add_argument('-sfn','--skip-first-n',action='store',type=int,default=0)
        parser.add_argument('--do-not-log-transform',action='store_true',default=False)
        parser.add_argument('--subtract-blanks',action='store_true',default=False)
        parser.add_argument('--subtract-control',action='store_true',default=False)
        parser.add_argument('--keep-missing-time-points',action='store_true',default=False)
        parser.add_argument('--verbose',action='store_true',default=False)
        parser.add_argument('--plot',action='store_true',default=False)
        parser.add_argument('--plot-derivative',action='store_true',default=False)
        parser.add_argument('--pool-by',required=False)
        parser.add_argument('--save-cleaned-data',action='store_true',default=False)
        parser.add_argument('--save-mapping-tables',action='store_true',default=False)
        parser.add_argument('--save-gp-data',action='store_true',default=False)
        parser.add_argument('--merge-summary',action='store_true',default=False)
        parser.add_argument('--fix-noise',action='store_true',default=False)
        parser.add_argument('--sample-posterior',action='store_true',default=False)

        if len(sys.argv) == 2:
            parser.print_help()
            sys.exit(0)

        args = parser.parse_args(sys.argv[2:])

        if (args.fix_noise) and (not args.pool_by):
            msg = '\nWARNING: --fix-noise and is only applicable if user also '
            msg += 'requests pooling with the --pool-by argument.'
            print(msg)

        if (args.sample_posterior) and (not args.pool_by):
            msg = '\nWARNING: --sample-posterior is only applicable if user also '
            msg += 'requests pooling with the --pool-by argument.'
            print(msg)

        if args.do_not_log_transform:
            args.log_transform = False
        else:
            args.log_transform = True

        if args.subset:
            args.merges = True

        if args.verbose:
            print_arguments(args)

        # unnecessary args
        args.hypothesis = None

        Command(args).fit()


    def heatmap(self):

        parser = argparse.ArgumentParser(
            description='Plot a heatmap')
        parser.add_argument('-i','--input',required=True)
        parser.add_argument('-o','--output',required=True)
        parser.add_argument('-s','--subset',required=False)
        parser.add_argument('-v','--value',required=True)
        parser.add_argument('-x','--x-variable',required=True)
        parser.add_argument('-y','--y-variable',required=True)
        parser.add_argument('-p','--operation',required=False,default='mean',
            choices=['mean','median'])
        parser.add_argument('-f','--filter',required=False)
        parser.add_argument('-t','--title',required=False)
        parser.add_argument('--kwargs',required=False)
        parser.add_argument('--verbose',action='store_true',default=False)
        parser.add_argument('--save-filtered-table',action='store_true',default=False)

        # sizing 
        parser.add_argument('--width-height',required=False,nargs=2)
        parser.add_argument('--colorbar-ratio',required=False,default=0.1,type=float,
            help='Proportion of figure size devoted to color bar. Default is 0.1')

        # coloring labels
        parser.add_argument('--color-x-by',required=False)
        parser.add_argument('--color-y-by',required=False)
        parser.add_argument('--color-file-x',required=False)
        parser.add_argument('--color-file-y',required=False)
        parser.add_argument('--color-scheme-x',required=False)
        parser.add_argument('--color-scheme-y',required=False)
        parser.add_argument('--color-x-ratio',required=False,default=0.1,type=float,
            help='Proportion of the heatmap devoted to the column color labels. Default is 0.1')
        parser.add_argument('--color-y-ratio',required=False,default=0.1,type=float,
            help='Proportion of the heatmap devoted to the row color labels. Default is 0.1')
        parser.add_argument('--missing-color',required=False,default=None)

        # sorting
        parser.add_argument('--cluster-x',required=False,default=False,action='store_true')
        parser.add_argument('--cluster-y',required=False,default=False,action='store_true')     
        parser.add_argument('--sort-x-by',required=False)
        parser.add_argument('--sort-y-by',required=False)

        # handling missing data
        parser.add_argument('--keep-rows-missing-data',action='store_true',default=False,
            help='Drops columsn that have any missing data')
        parser.add_argument('--keep-columns-missing-data',action='store_true',default=False,
            help='Drops rows that have any missing data')

        # handling label sizes and adjusting tick labels
        parser.add_argument('--x-tick-labels-scale',required=False,default=config['x_tick_labels_scale'],
            help='Must be between 0 (smallest) and 1 (largest).')
        parser.add_argument('--y-tick-labels-scale',required=False,default=config['y_tick_labels_scale'],
            help='Must be between 0 (smallest) and 1 (largest).')
        parser.add_argument('--color-bar-labels-scale',required=False,default=config['color_bar_labels_scale'],
            help='Must be between 0 (smallest) and 1 (largest).')
        parser.add_argument('--x-rotation',required=False,default=90)
        parser.add_argument('--highlight-labels',required=False)

        if len(sys.argv) == 2:
            parser.print_help()
            sys.exit(0)

        args = parser.parse_args(sys.argv[2:])

        if args.verbose:
            print_arguments(args)
            
        from amiga.libs.heatmap import main as heatmap_main
        
        heatmap_main(args)


    def normalize(self):
        parser = argparse.ArgumentParser(
            description='Compare two growth curves')

        parser.add_argument('-i','--input',required=True)
        parser.add_argument('--over-write',action='store_true',default=False,
            help='Over-write file otherwise a new copy is made with "_normalize" suffix')
        parser.add_argument('--verbose',action='store_true',default=False)
        parser.add_argument('--group-by',required=False)
        parser.add_argument('--normalize-by',required=False)
        parser.add_argument('--normalize-method',action='store',default='subtraction',
            choices=['division','subtraction'])

        if len(sys.argv) == 2:
            parser.print_help()
            sys.exit(0)

        args = parser.parse_args(sys.argv[2:])

        if args.verbose:
            print_arguments(args)

        from amiga.libs.normalize import main as normalize_main

        normalize_main(args)


    def print_defaults(self):

        msg = '\nDefault settings for select variables. '
        msg += 'You can adjust these values in "amiga/libs/config.py". \n\n'
        msg += tidyDictPrint(config)
        print(msg)
        #sys.exit(msg)


    def summarize(self):
        parser = argparse.ArgumentParser(
            description='Perform a basic summary and plot curves')

        parser.add_argument('-i','--input',required=True)
        parser.add_argument('-o','--output',required=False)
        parser.add_argument('--dont-plot',action='store_true',default=False)
        parser.add_argument('--merge-summary',action='store_true',default=False)
        parser.add_argument('--verbose',action='store_true',default=False)
        parser.add_argument('-f','--flag',required=False)
        parser.add_argument('-s','--subset',required=False)
        parser.add_argument('-y','--hypothesis',required=False)
        parser.add_argument('-t','--interval',required=False)
        parser.add_argument('--save-cleaned-data',action='store_true',default=False)
        parser.add_argument('--save-mapping-tables',action='store_true',default=False)
        parser.add_argument('--subtract-blanks',action='store_true',default=False)
        parser.add_argument('--subtract-control',action='store_true',default=False)

        if len(sys.argv) == 2:
            parser.print_help()
            sys.exit(0)

        args = parser.parse_args(sys.argv[2:])

        args.subset = None  ## subsetting is not implemented
        args.hypothesis = None ## hypothis are not implemented

        if args.verbose:
            print_arguments(args)

        Command(args).summarize()


    def test(self):
        parser = argparse.ArgumentParser(
            description='Test for differential growth between two conditions')

        # defining input/output
        parser.add_argument('-i','--input',required=True)
        parser.add_argument('-o','--output',required=False)
        parser.add_argument('-f','--flag',required=False)
        parser.add_argument('-s','--subset',required=False)
        parser.add_argument('-t','--interval',required=False)
        parser.add_argument('-y','--hypothesis',required=True)
        parser.add_argument('-sfn','--skip-first-n',action='store',type=int,default=0)
        parser.add_argument('-tss','--time-step-size',action='store',type=int,default=1)#11
        parser.add_argument('-np','--number-permutations',action='store',type=int,default=0)
        parser.add_argument('-fdr','--false-discovery-rate',action='store',type=int,default=10)
        parser.add_argument('--confidence',required=False,type=float,default=95,
            help='Must be between 80 and 100. Default is 95.')
        parser.add_argument('--subtract-blanks',action='store_true',default=False)
        parser.add_argument('--subtract-control',action='store_true',default=False)
        parser.add_argument('--verbose',action='store_true',default=False)
        parser.add_argument('--fix-noise',action='store_true',default=False)
        parser.add_argument('--include-gaussian-noise',action='store_true',default=False)
        parser.add_argument('--sample-posterior',action='store_true',default=False)
        parser.add_argument('--dont-plot',action='store_true',default=False)
        parser.add_argument('--dont-plot-delta-od',action='store_true',default=True)
        parser.add_argument('--save-cleaned-data',action='store_true',default=False)
        parser.add_argument('--save-mapping-tables',action='store_true',default=False)
        parser.add_argument('--save-gp-data',action='store_true',default=False)
        parser.add_argument('--merge-summary',action='store_true',default=False)
        parser.add_argument('--do-not-log-transform',action='store_true',default=True)

        if len(sys.argv) == 2:
            parser.print_help()
            sys.exit(0)

        args = parser.parse_args(sys.argv[2:])

        if (args.confidence < 80) | (args.confidence > 100):
            msg = 'FATAL USER ERROR: Confdience must be between 80 and 100.'
            sys.exit(msg)

        if args.do_not_log_transform:
            args.log_transform = False
        else:
            args.log_transform = True
        
        args.confidence = args.confidence / 100

        if args.verbose:
            print_arguments(args)

        Command(args).test()

if __name__ == '__main__':

    AMiGA()
