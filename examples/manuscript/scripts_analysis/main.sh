#!/usr/bin/bash

# __author__ = Firas S Midani
# __email__ = midani@bcm.edu

# Midani et al. (2020) driver script for analyzing data. 

# first argument is the file path that activates your python virtual environment
venv=$1

# second argument is the file path to the amiga directory
amiga=$2

sh $samiga/example/manuscript/scripts_analysis/analyze_CD1007_Fructose.sh  $venv  $amiga
sh $samiga/example/manuscript/scripts_analysis/analyze_CD2058_Glucose.sh  $venv  $amiga
sh $samiga/example/manuscript/scripts_analysis/analyze_biolog_data.sh  $venv  $amiga
sh $samiga/example/manuscript/scripts_analysis/analyze_death_experiment.sh  $venv  $amiga
sh $samiga/example/manuscript/scripts_analysis/test_death_experiment_hypotheses.sh  $venv  $amiga