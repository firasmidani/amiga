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

# default wavelength for OD measurements is et to 620 nm; str because it is used only for plot labeling
config['nm'] = str(620)

# parameters related to plotting and fold change
config['fcg'] = 1.50  # fold-change threshold for growth
config['fcd'] = 0.50  # fold-change threshold for death

config['fcg_line_color'] = (0.0,0.0,1.0,1.0)
config['fcg_face_color'] = (0.0,0.0,1.0,0.15)

config['fcd_line_color'] = (1.0,0.0,0.0,1.0)
config['fcd_face_color'] = (1.0,0.0,0.0,0.15)

config['fcn_line_color'] = (0.0,0.0,0.0,1.0)
config['fcn_face_color'] = (0.0,0.0,0.0,0.15)

# parameters related to annotating grid plots with OD Max and Well ID values
config['fcn_well_id_color'] = (0.65,0.165,0.16,0.8)
config['fcn_od_max_color'] = (0.0,0.0,0.0,1.0)

# parameter for labeling y-axis of grid plots
config['grid_plot_y_label'] = 'Optical Density (620 nm)'

# diaxuic shift paramters
config['diauxie'] = 0.333  # minimum ratio relative to maximum peak for each growth phase
