#!/usr/bin/bash

# __author__ = Firas S Midani
# __email__ = midani@bcm.edu

# Midani et al. (2020) driver script for generating figures.

# first argument is the file path that activates your python virtual environment
source $1

# second argument is the file path to the amiga directory
amiga=$2

# check that a folder exists for storing PDFs, else create.
mkdir -p $amiga/examples/manuscript/figures/

python $amiga/examples/manuscript/scripts_figures/figure_1_ab.py
python $amiga/examples/manuscript/scripts_figures/figure_1_cd.py
python $amiga/examples/manuscript/scripts_figures/figure_2.py
python $amiga/examples/manuscript/scripts_figures/figure_3.py
python $amiga/examples/manuscript/scripts_figures/figure_4.py

python $amiga/examples/manuscript/scripts_figures/supp_figure_1.py
python $amiga/examples/manuscript/scripts_figures/supp_figure_4.py
python $amiga/examples/manuscript/scripts_figures/supp_figure_5.py
python $amiga/examples/manuscript/scripts_figures/supp_figure_6.py
python $amiga/examples/manuscript/scripts_figures/supp_figure_7.py

# supp figure_1 (need to add text in powerpoint)

# supp figure_2 (simply copy and rename)
cp $amiga/examples/manuscript/biolog/figures/CD2015_PM1-1_fit.pdf $amiga/examples/manuscript/figures/Midani_AMiGA_Supp_Figure_2.pdf

# supp figure_3 (stitch two heatmaps in powerpoint and add panel letters)
cp $amiga/examples/manuscript/biolog/summary/CD2015_split_norm_auc_log.pdf $amiga/examples/manuscript/figures/Midani_AMiGA_Supp_Figure_3A.pdf
cp $amiga/examples/manuscript/biolog/summary/CD2015_pooled_norm_auc_log.pdf $amiga/examples/manuscript/figures/Midani_AMiGA_Supp_Figure_3B.pdf