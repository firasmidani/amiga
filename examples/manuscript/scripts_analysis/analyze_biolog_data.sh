#!/bin/bash

# __author__ = Firas S Midani
# __email__ = midani@bcm.edu

# Midani et al. (2021) Figures 1 & 2, Supp. Figures 1-4 use data generated by this script 

# load amiga source environment
source $1

# create variable for the absolute file path to amiga.py
amiga=$2

# create variable for the absolute file path to the working directory
input="${amiga}/examples/manuscript/biolog"

# perform a baisc summary of data and plot 96-well plates
python $amiga/amiga.py summarize -i $input --save-cleaned-data --save-mapping-tables --verbose 

# infer growth parameters with GP regression on each individual growth curve
python $amiga/amiga.py fit -i $input -o 'split' \
						   --merge-summary --save-gp-data  \
						   --plot --plot-derivative \
					       --skip-first-n 1 -tss 1 \
					       --verbose  

# infer growth parameters with GP regression on pooled replicates
python $amiga/amiga.py fit -i $input -o 'pooled_by_isolate' \
					       --pool-by 'Isolate,Substrate,PM' \
					       --skip-first-n 1 -tss 1  \
					       --save-gp-data --verbose

# normalize split growth parameters by divison
python $amiga/amiga.py normalize -i $input/summary/split_summary.txt \
								 --normalize-method 'division' \
								 --group-by 'Plate_ID' \
								 --normalize-by 'Substrate:Negative Control'

# normalize pooled growth parameters by divison
python $amiga/amiga.py normalize -i $input/summary/pooled_by_isolate_summary.txt \
								 --normalize-method 'division' \
								 --group-by 'Isolate,PM' \
								 --normalize-by 'Substrate:Negative Control'

# plot summary heatmaps

## split normalized (in order: Norm. AUC (log), Norm. AUC (lin), Norm. Growth Rate)
python $amiga/amiga.py heatmap -i $input/summary/split_summary_normalized.txt \
                         	   -o CD2015_split_norm_auc_log \
					    	   -x Plate_ID -y Substrate -v 'norm(auc_log)' \
					     	   -f 'row any >= 1.2 OR row any = 1 OR row any <= 0.8' \
					     	   --kwargs 'center:1;vmin:0;row_cluster:True;annot:True' \
					     	   --title 'Normalized Area Under the Curve' --verbose

python $amiga/amiga.py heatmap -i $input/summary/split_summary_normalized.txt \
                         	   -o CD2015_split_norm_auc_lin \
					    	   -x Plate_ID -y Substrate -v 'norm(auc_lin)' \
					     	   -f 'row any >= 1.2 OR row any = 1 OR row any <= 0.8' \
					     	   --kwargs 'center:1;vmin:0;row_cluster:True;annot:True' \
					     	   --title 'Normalized Area Under the Curve' --verbose

python $amiga/amiga.py heatmap -i $input/summary/split_summary_normalized.txt \
                         	   -o CD2015_split_norm_gr \
					    	   -x Plate_ID -y Substrate -v 'norm(gr)' \
					     	   -f 'row any >= 1.5 OR row any = 1 OR row any <= 0.5' \
					     	   --kwargs 'center:1;vmin:0;row_cluster:True;annot:True' \
					     	   --title 'Normalized Area Under the Curve' --verbose

## pooled normalized (in order: Norm. AUC (log), Norm. AUC (lin), Norm. Growth Rate)
python $amiga/amiga.py heatmap -i $input/summary/pooled_by_isolate_summary_normalized.txt \
                         	   -o CD2015_pooled_norm_auc_log	 \
					    	   -x Isolate -y Substrate -v 'norm(auc_log)' \
					     	   -f 'row any >= 1.2 OR row any = 1 OR row any <= 0.8' \
					     	   --kwargs 'center:1;vmin:0;row_cluster:True;annot:True' \
					     	   --title 'Normalized Area Under the Curve' --verbose

python $amiga/amiga.py heatmap -i $input/summary/pooled_by_isolate_summary_normalized.txt \
                         	   -o CD2015_pooled_norm_gr \
					    	   -x Isolate -y Substrate -v 'norm(gr)' \
					     	   -f 'row any >= 1.5 OR row any = 1 OR row any <= 0.5' \
					     	   --kwargs 'center:1;vmin:0;row_cluster:True;annot:True' \
					     	   --title 'Normalized Area Under the Curve' --verbose
