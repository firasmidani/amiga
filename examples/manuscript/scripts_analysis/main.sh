#!/usr/bin/bash

# __author__ = Firas S Midani
# __email__ = midani@bcm.edu

# Midani et al. (2021) driver script for analyzing data. 

# first argument is the file path that activates your python virtual environment
venv=$1

# second argument is the file path to the amiga directory
amiga=$2

# # analysis for figure 1A-B
# sh $amiga/examples/manuscript/scripts_analysis/analyze_CD1007_Fructose.sh  $venv  $amiga

# # analysis for figure 1C-D
# sh $amiga/examples/manuscript/scripts_analysis/analyze_CD2058_Glucose.sh  $venv  $amiga

# # analysis for figure 2, and supplementary figures 1, 2, 3, and 4
# sh $amiga/examples/manuscript/scripts_analysis/analyze_biolog_data.sh  $venv  $amiga

# # analysis for figure 3 and 4, and supplementary figure 6
# sh $amiga/examples/manuscript/scripts_analysis/analyze_death_experiment.sh  $venv  $amiga

# # analysis for figure 4
# sh $amiga/examples/manuscript/scripts_analysis/test_death_experiment_hypotheses.sh  $venv  $amiga

# analysis for supplementary figures 5 and 7
sh $amiga/examples/manuscript/scripts_analysis/analyze_non_c_diff_examples.sh  $venv  $amiga