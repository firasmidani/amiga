---
layout: page
title: "Command Line Interface"
category: doc
date: 2020-04-16 21:42:08
order: 14
use_math: false
---
<!-- AMiGA is covered under the GPL-3 license -->

To see the full list of arguments that `AMiGA` will accept, you can pass it the `-h` or `--help` argument.

```bash
python amiga.py --help
```

which will reveal the following message

```bash
usage: amiga.py [-h] -i INPUT [-o OUTPUT] [-f FLAG] [-s SUBSET]
                [-y HYPOTHESIS] [-t INTERVAL] [-p] [-v]
                [-np NUMBER_PERMUTATIONS] [-tss TIME_STEP_SIZE]
                [-sfn SKIP_FIRST_N] [-fdr FALSE_DISCOVERY_RATE]
                [--merge-summary] [--normalize-parameters] [--pool-by POOL_BY]
                [--normalize-by NORMALIZE_BY] [--plot-derivative]
                [--plot-delta-od] [--only-basic-summary] [--save-cleaned-data]
                [--save-gp-data] [--save-mapping-tables]
                [--only-print-defaults] [--subtract-control]
                [--dont-plot] [--fix-noise] [--sample-posterior]
                [--include-gaussian-noise]

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
  -o OUTPUT, --output OUTPUT
  -f FLAG, --flag FLAG
  -s SUBSET, --subset SUBSET
  -y HYPOTHESIS, --hypothesis HYPOTHESIS
  -t INTERVAL, --interval INTERVAL
  -p, --plot
  -v, --verbose
  -np NUMBER_PERMUTATIONS, --number-permutations NUMBER_PERMUTATIONS
  -tss TIME_STEP_SIZE, --time-step-size TIME_STEP_SIZE
  -sfn SKIP_FIRST_N, --skip-first-n SKIP_FIRST_N
  -fdr FALSE_DISCOVERY_RATE, --false-discovery-rate FALSE_DISCOVERY_RATE
  --merge-summary
  --normalize-parameters
  --normalize-by NORMALIZE_BY
  --pool-by POOL_BY
  --plot-derivative
  --plot-delta-od
  --only-basic-summary
  --save-cleaned-data
  --save-gp-data
  --save-mapping-tables
  --only-print-defaults
  --subtract-control
  --dont-plot
  --fix-noise
  --sample-posterior
  --include-gaussian-noise
```

<br/>

**Selecting input/output**

`-i` or `--input`

Accepts a string which is the path to a working directory or a specific data file. The path can be relative or absolute.

`-o` or `--output`

Accepts a string which defines the filename (no extension and no path) for your summary or data files. If AMiGA needs to merge results across multiple plates (due to the `--subset` or `--merge-summary` arguments) but no filename is passed here, AMiGA will instead give files a unique time stamp (e.g. 2020-08-26_09-09-59).

<br/>

**Reducing samples**

`-f` or `--flag`

Defines which wells should be excluded form the analysis. See [Data Subsetting](/amiga/doc/subsetting.html) for more details.

`-s` or `--subset`

Defines which files to be included in the analysis. See [Data Subsetting](/amiga/doc/subsetting.html) for more details.

<br/>

**Reducing time points**

`-t` or `--interval`

Used to define the interval between time points in the data passed to `AMiGA`. The default is 600 seconds. The default can be also modified in the `libs/config.py` file. In addition, user can modify the unit of time in the data input and output.

`-sfn` or `--skip-first-n`

Accepts an integer which is the number of time points to ignore in the beginnning. Default is 0. 

`-tss` or `--time-step-size`

Accepts an integer which is used to define how many time points to include in the GP model. 1 indicates that each time point is included (i.e. T0, T1, T2, ...); 2 indicates that every other time point is included beginning with the first one (i.e. T0, T2, T4, ...); and so on. The default value is 1. Thinning of the data helps speed up the GP inference and reduction of the input to the GP model does not drastically alter overall growth curve shape as long as the input growth curve is smooth and does not exhibit acute spikes or dips. However, thinning of the data may alter estimates of growth parameters that describe fast dynamics (e.g. lag time and adaptation time) so verify that your analysis would not be affected by reducing time points before doing so. 

<br/>

**Hypothesis testing**

`-y` or `--hypothesis`

Defines the hypothesis for GP regression model to be tested by AMiGA on the user-passed data. See [Hypothesis Testing](/amiga/doc/hypothesistesting.html) for more details.

`-np` or --number-permutations`

Accepts an integer and defines how many permutations to perform for estimating the null distribution of the Bayes Factor. See [Hypothesis Testing](/amiga/doc/hypothesistesting.html) for more details. The default value is 20.

`-fdr` or --false-discovery-rate`

Accepts an integer and defines the False Discovery Rate (FDR) threshold used in hypothesis testing. The default value is 10. See [Hypothesis Testing](/amiga/doc/hypothesistesting.html) for more details.  

`--subtract-control`

A boolean argument on whether to condition samples on controls (i.e. subtract control growth curve from treatment growth curve) before testing for differences in growth using GP inference. See [Hypothesis Testing](/amiga/doc/hypothesistesting.html) for more details.

<br/>

**Pooling and normalization**

`--normalize-parameters`

A boolean argument that will normalize growth parameters of each well in a plate relative to its group-specific control sample(s).See [Normalizing Parameters](/amiga/doc/normalizing.html) for more details.

`--pool-by`

Accepts comma-separated list of variables in mapping files that will be used to specify how samples will be grouped See [Pooling](/amiga/doc/pooling.html) for more details.

`--normalize-by`

Accepts comma-separated list of variables in mapping files that will be used to specify which wells are control samples. Only used for analysis with pooled replicates. See [Normalizing Parameters](/amiga/doc/normalizing.html) for more details.

<br/>

**Plotting**

`-p` or `--plot`

A boolean argument. If invoked, `AMiGA` will plot 96-well plates and overlay actual and model-predicted growth curves.

`--plot-derivative`

A boolean argument that would save grid plots for 96-well plates of the estimated rate of change of OD in each well. Values above zero correspond to positive rate of change (i.e. growth or increase in OD) and values below zero correspond to negative rate of change (i.e. death or reduction in OD). See [Summarizing & Plotting Data](/amiga/doc/plotting.html) for more details on the colors in plots.

`--plot-delta-od`

A boolean argument that would compute and plot the functional OD difference between two growth curves compared for hypothesis testing. See [Hypothesis Testing](/amiga/doc/hypothesistesting.html) for more details.  

`--dont-plot`

If you would like to run a basic summary or growth fitting (i.e. **\-\-only-basic-summary**) of the data (no GP modelling) without plotting figures, you should pass this argument.

<br/>

**saving input/output**

`--merge-summary`

Instead of plate-specific summary files, a single summary file will be reported in the `summary` folder. This will occur by default if user requests any subsetting of the data.

`--save-cleaned-data`

A boolean argument that will save the data OD extracted from the files provided in the `data` folder. These derived data files will not include any meta-data and will be formatted as time x samples and saved in the `derived` folder. Column headers corresponds to the `Well` column in the corresponding key files saved in `summary` folder.

`--save-gp-data`

A boolean argument that will save the different variants of the growth curve OD data before and after it has been fit with Gaussian Process model. Data will be formatted in a long-format where each row is a specific time point for a specific well in a specific plate. Identifying columns include `Time`,`Sample_ID`,`Plate_ID`. The `Plate_ID` corresponds to the name of the file from which the growth curve was processed. The `Sample_ID` corresponds to a specific well, see corresponding key files saved in `summary` for which well. Data columns include the following:
* `Time`: time points. 
* `GP_Input`: OD after log-transformation and baseline (OD at T0) subtraction. This is the OD variant explicitly modeled by a GP. OD at all time points will be included here even if user requested thinning of the data during modelling. 
* `GP_Output`: Predicted OD by the GP model based on the thinned `GP_Input`.
* `GP_Derivative`: Predicted derivative of the OD (dOD/dt) by the GP model.
* `OD_Growth_Data`: `GP_Input` that has been converted back to real OD but still baseline-corrected.
* `OD_Growth_Fit`: `GP_Output` that has ben converted back real OD but still baseline-corrected.
* `OD_Fit`: Predicted fit for the original OD. This is essentially `OD_Growth_Fit` but with baseline (OD at T0) added back.

`--save-mapping-tables`

A boolean argument that will ask force AMiGA to save all internally-generated or -parsed mapping files into `mapping` folder.

<br/>

**User communication**

`--only-print-defaults`

A boolean argument that will print to the command terminal the parameters defined in the `libs/config.py` file. You can replace these default parameters to suit your analysis.

`-v` or `--verbose`

A boolean argument. If invoked, `AMiGA` will communicate via the command terminal with the user by sharing in real time more details about the input, data processing, and results. This is extremely helpful for troubleshooting. 

<br/>

**Setting analysis preferences**

`--only-basic-summary`

See [Summarizing & Plotting Data](/amiga/doc/plotting.html) for more details.

`--fix-noise`

A boolean parameter that dictates whether timepoint-specific noise should be included in the GP model. A GP model inherently models noise (error) as a single hyperparameter because it assumes that the noise is the same across all time points. This may not be the case for all growth data. Here, uers can opt for `AMiGA` to estimate the variance of the data (in pooled growth fitting) and pass those estiamtes of noise to the GP model which will fix the noise at each time point according to the pre-computed variance. Results seldom vary between standad GP model and fixed-noise GP models. However, the latter is more prone to overfitting and is computationally slower.  

`--sample-posterior`

A boolean argument on whether to infer summary statistics for all growth parameters. If not called, `AMiGA` povides a single estimate of each growth parameter based on the mean latent function of the GP model. If called, `AMiGA` samples from the posteiror distribution (`n` times, where `n` is defeined in the `libs/config.py` file), compute the growth parameter for each sampled curve, then returns the mean and the standard deivation of the mean of each growth parameter. 
 
`--include-guassian-noise`

A boolean argument on whether plotted credible intervals should include estiamted Guassian noise in addition to the credible interval of the posterior of the latent function. This currently is only applicable for hypothesis testing plots. If called in conjunction with `\-\-fix-noise` the credible interval will likely vary over time. 

