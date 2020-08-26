#!/bin/bash

# __author__ = Firas S Midani
# __email__ = midani@bcm.edu

## load amiga source environment
source /Users/firasmidani/rab_fm/envs/python/AMIGA/bin/activate

## define conditions of interest
declare -a RIBOTYPES=("RT027" "RT078" "RT053" "RT001")
declare -a SUBSTRATES=("Glucose" "Fructose")
declare -a CONCENTRATIONS=("Low" "High")

## define common command parameters
BASE="python /Users/firasmidani/rab_fm/git/amiga/amiga.py"   # path to amiga
INPUT="/Users/firasmidani/rab_fm/docs/amiga/analysis/death"  # path to working directory
HYPO="'H0:Time;H1:Time+Concentration'"						 # hypothesis
PARAMS="--verbose -np 100 -tss 10 -fdr 10 --plot --plot-delta-od --dont-subtract-control" 

RESULTS=()  # array to maintain list of results files

## assemble and call unique commands
for st in "${SUBSTRATES[@]}"
do
	for rt in "${RIBOTYPES[@]}"
	do 
			# assemble condition-specific arguments
			SUBSET="'Substrate:${st};Ribotype:${rt};Concentration:Low,High'"
			OUTPUT="${st}_LH_${rt}_y_c"

			# assemble AMiGA command
			CMD="${BASE} -i ${INPUT} -s ${SUBSET} -y ${HYPO} -o ${OUTPUT} ${PARAMS}"

			# print command to terminal and execute
			echo $CMD
			eval $CMD

			# maintain list of results files
			RESULT="${INPUT}/models/${OUTPUT}/${OUTPUT}_log.txt"
			RESULTS+=("${RESULT}")
	done
done

## create summary text file, if it already exists, empty it.
LOG="/Users/firasmidani/rab_fm/docs/amiga/analysis/death/models/models_summary.txt"
echo -n "" > $LOG

## merge tables from all results files, with a single header
awk 'FNR==1 && NR!=1 {next;}{print}' "${RESULTS[@]}" > $LOG

## end