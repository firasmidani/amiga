# Analysis and Figures for AMiGA Manuscript

**AMiGA: Software for Automated Analysis of Mirobial Growth Assays**
FS Midani, J Collins, and RA Britton. *bioRxiv* (2020). 

The `./amiga/examples/manuscript/` folder houses all of the data and code needed to generate the fgures for the manuscript. To re-run the analysis and generate the figures on your machine, please do the following:

1. Make sure that AMiGA is installed and runs successfully on your machine. See [Installation](https://firasmidani.github.io/amiga/doc/installation.html) for instructions and requirements.

2. Change directoy into`./amiga/examples/manuscript/`

3. In your shell terminal, define variables for (i) file path that activates your python virtual environment and (ii) your current installation of `AMiGA`. For example,

```bash
venv=/Users/firasmidani/rab_fm/envs/python/AMIGA/bin/activate
amiga=/Users/firasmidani/rab_fm/git/amiga
```

3. Using your terminal, execute the analysis by passing the above variables to the main driver script for analysis. 

```bash
sh ./scripts_analysis/main.sh $venv $amiga
```

This step will take approximately 30 minutes but may depend on the speed/memory of your machine. You can speed the up analysis by increasing the `time step size` using the `-tss` argument or reducing the number of permutations for hypothesis testing. See [Command Line Interface](https://firasmidani.github.io/amiga/doc/command-line-interface.html) for more details; however, keep in mind that increasing the time step size will affect estimates of growth parameters because you are reducing the input or number of time points used by model. To increase the `time step size`, you can edit the commands in the scripts directly. For example, adjust the `-tss` arguments in lines 8, 25, and 32 of `analyze_death_experiment.sh`. To adjust the number of test permutations, you only need to adjust the `-np` argument in line 23 of `test_death_experiment_hypotheses.sh`.

4. Using your temrinal, excute the figure generation by passing the above second variable to the main driver script for figure generation. Figures will be saved as PDFs in the `./amiga/examples/manuscript/scripts_figues` folder.

```bash
sh ./scripts_figures/main.sh $venv $amiga
```

