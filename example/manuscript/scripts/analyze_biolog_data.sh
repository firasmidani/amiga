#!/bin/bash

# __author__ = Firas S Midani
# __email__ = midani@bcm.edu


# load amiga source environment
source /Users/firasmidani/rab_fm/envs/python/AMIGA/bin/activate

# create variable for the absolute file path to amiga.py
amiga="/Users/firasmidani/rab_fm/git/amiga"

# create variable for the absolute file path to the working directory
input="/Users/firasmidani/rab_fm/docs/amiga/analysis/biolog"

# perform a baisc summary of data and plot 96-well plates
python $amiga/amiga.py -i $input --only-basic-summary \
					   --save-cleaned-data --save-mapping-tables --verbose 

# infer growth parameters with GP regression on each individual growth curve
python $amiga/amiga.py -i $input -o 'split_normalized_merged' \
					     --normalize-parameters --merge-summary --save-gp-data  \
					     --plot --plot-derivative \
					     --skip-first-n 1 -tss 4 \
					     --verbose  

# infer growth parameters with GP regression on pooled replicates
python $amiga/amiga.py -i $input -o 'pooled_normalized_merged_by_ribotype' \
					   --pool-by 'Isolate,Substrate,PM' \
					   --normalize-by 'Substrate:Negative Control' \
					   --skip-first-n 1 -tss 4 --sample-posterior  \
					   --save-gp-data --verbose

#plot summary heatmaps

python $amiga/heatmap.py -i $input/summary/split_normalized_merged_summary.txt \
                         -o CD2015_split_norm_auc \
					     -x Plate_ID -y Substrate -v 'norm(auc_log)' \
					     -f 'row any >= 1.2 OR row any = 1 OR row any <= 0.8' \
					     --kwargs 'center:1;vmin:0' \
					     --title 'Normalized Area Under the Curve' --verbose

python $amiga/heatmap.py -i $input/summary/split_normalized_merged_summary.txt \
                         -o CD2015_split_norm_gr \
					     -x Plate_ID -y Substrate -v 'norm(gr)' \
					     -f 'row any >= 1.5 OR row any = 1 OR row any <= 0.5' \
					     --kwargs 'center:1;vmin:0' \
					     --title 'Normalized Growth Rate' --verbose

python $amiga/heatmap.py -i $input/summary/pooled_normalized_merged_by_ribotype_summary.txt \
                         -o CD2015_pooled_norm_auc \
					     -x Isolate -y Substrate -v 'norm(auc_log)' \
					     -f 'row any >= 1.2 OR row any = 1 OR row any <= 0.8' \
					     --kwargs 'center:1;vmin:0' \
					     --title 'Normalized Area Under the Curve' --verbose

python $amiga/heatmap.py -i $input/summary/pooled_normalized_merged_by_ribotype_summary.txt \
                         -o CD2015_pooled_norm_gr \
					     -x Isolate -y Substrate -v 'norm(gr)' \
					     -f 'row any >= 1.5 OR row any = 1 OR row any <= 0.5' \
					     --kwargs 'center:1;vmin:0' \
					     --title 'Normalized Growth Rate' --verbose
