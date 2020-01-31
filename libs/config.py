#!/usr/bin/env python

'''
DESCRIPTION configuration file for setting default parameters used by AMiGA
'''

__author__ = "Firas Said Midani"
__version__  = "0.1.0"
__email__ = "midani@bcm.edu"

config = {}

# default time interval between OD measurments is set to 600 seconds
config['interval'] = 600

# default wavelength for OD measurements is et to 620 nm, used for plot labeling only
config['nm'] = 620
