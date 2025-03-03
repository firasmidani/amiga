#!/usr/bin/env python

'''
AMiGA configuration file for setting and modifying default parameters used by AMiGA.
'''

__author__ = "Firas S Midani"
__email__ = "midani@bcm.edu"


# NOTES
#     colors are defined in (R,G,B,A) format where A is alpha and all values range
#     	 from 0.0 to 1.0 (i.e. map 0 to 255) but you can define colors in text or hex
#        string format, see https://het.as.utexas.edu/HET/Software/Matplotlib/api/colors_api.html 

config = {}

###	--------------------- ###
### DATA INPUT PARAMETERS ###
###	--------------------- ###

# acceptable values are 'seconds', 'minutes', or 'hours'
config['time_input_unit'] = 'seconds'
config['time_output_unit'] = 'hours'

# default time interval between OD measurments is set to 600 seconds
config['interval'] = 600  # units ared based on 'time_input_unit' above  

# whether to estimate OD at the first time point using a polynomial regression fit across replicates
config['PolyFit'] = False

# How to handle nonpositive (i.e. zero or negative) values. See AMiGA docs for more details
config['handling_nonpositives'] = 'Delta' # 'Delta' or 'LOD'

config['limit_of_detection'] = 0.010 # must be numeric and positive (i.e. not zero)
config['force_limit_of_detection'] = False # or True. Only applies if: config['handling_nonpositives'] = 'Delta'
config['number_of_deltas'] = 5 # must be integer greater than 1, can be really large number (e.g. 1000) to comply with curves of various lengths
config['choice_of_deltas'] = 'median' # or min or max or mean

# Whether to drop blank or control wells?
config['drop_blank_wells'] = False
config['drop_control_wells'] = False

###	------------- ###
### 96-Well Plots ###
###	------------- ###

# parameters related to plotting and fold change
config['fcg'] = 1.5  # fold-change threshold for growth
config['fcd'] = 0.5  # fold-change threshold for death

config['fcg_line_color'] = (0.0,0.0,1.0,1.0)
config['fcg_face_color'] = (0.0,0.0,1.0,0.15)

config['fcd_line_color'] = (1.0,0.0,0.0,1.0)
config['fcd_face_color'] = (1.0,0.0,0.0,0.15)

config['fcn_line_color'] = (0.0,0.0,0.0,1.0)  # fc-neutral: i.e. fold-change is within thresholds defined above
config['fcn_face_color'] = (0.0,0.0,0.0,0.15)  # fc-neutral: i.e. fold-change is within thresholds defined above

config['gp_line_fit'] = 'yellow'

# parameters related to annotating grid plots with OD Max and Well ID values
config['fcn_well_id_color'] = (0.65,0.165,0.16,0.8)
config['fcn_od_max_color'] = (0.0,0.0,0.0,1.0)

# parameter for labeling y-axis of grid plots
config['grid_plot_y_label'] = 'Optical Density'

# how to handle flagged wells in plots
config['plot_flag_wells'] = 'cross' # 'empty', cross', or 'keep'

### --------- ###
### Heat Maps ###
### --------- ###

config['x_tick_labels_scale'] = 1
config['y_tick_labels_scale'] = 1
config['color_bar_labels_scale'] = 0.2

###	---------------- ###
### Model Parameters ###
###	---------------- ###

# select mode for initializing hyperparamters of GP regression
#     1. "default" will set viarance and lengthscale parameters to 1
#     2. "moments-based-proper" will initialize lengthscale to range of time and
#            will optimize variance based on a grid search
#     3. "moments-based-fast" will initilize lengthscale to range of time and will
#            initilize variance to variance of measurements
config['initialization_method'] = 'default'  # "deault", "moments-based-proper" or "moments-based-fast"

# for GP regression with input-dependent noise, select a variance smoothing window: 
config['variance_smoothing_window'] = 6 # number of x-values, based on default paramters: 6 * 600 seconds = 1 hour

# for GP regression on individual curves, AMiGA can check quality of fit by comparing the 
# estiamted carrying capacity to the actual carrying capacity calculated from data
# K_actual = OD_Max - OD_Baseline or Adj_OD_Max-Adj_OD_Basline
# here, the user can define the threshold at which AMiGA will flag poor fit of a curve. the default
# threshold is 20%.
config['k_error_threshold'] = 20

###	------------------------ ###
### Hypothesis Testing Plots ###
###	------------------------ ###

# hypothesis testing plot colors (the first two are the default colors used by AMiGA)
config['hypo_colors']  = [(0.11,0.62,0.47),(0.85,0.37,0.01)]  # correspond to seagreen

config['hypo_plot_y_label'] = 'OD'

config['HypoPlotParams'] = {'overlay_actual_data':True,
							'fontsize':15,
							'tick_spacing':5,
							'legend':'outside'}

###	----------------- ###
### GROWTH PARAMETERS ###
###	----------------- ###

# adaptationtime
config['confidence_adapt_time'] = 0.95

# diaxuic shift paramters
config['diauxie_ratio_varb'] = 'K' # can be "K" (max OD) or "r" (max dOD/dt)
config['diauxie_ratio_min'] = 0.20  # minimum ratio relative to maximum growth or growth rate for each growth phase
config['diauxie_k_min'] = 0.10
 
# parameters for reporting results (not implemented yet)
config['report_parameters'] = ['auc_lin','auc_log','k_lin','k_log','death_lin','death_log',
							   'gr','dr','td','lagC','lagP',
                               't_k','t_gr','t_dr','diauxie']
#config['report_parameters'] = ['auc_lin','k_lin','lagP','x_dr','diauxie']

# how many samples from the posterior funtion are used for estimating mean/std of growth parameters
config['n_posterior_samples'] = 100

###	------------------ ###
### USER COMMUNICATION ###
###	------------------ ###

config['Ignore_RuntimeWarning'] = True
