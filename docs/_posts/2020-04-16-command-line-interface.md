---
layout: page
title: "Command Line Interface"
category: doc
date: 2020-04-16 21:42:08
order: 14
use_math: false
---
<!-- AMiGA is covered under the GPL-3 license -->

To see the full list of arguments that `AMiGA` will accept, you can pass the `-h` or `--help` to the command of interest which include `summarize`, `fit`, `normalize`, `test`, `heatmap`, `compare`, and `get-time`.

```bash
python amiga.py <command> --help
```

Below I elaborate on how to use all arguments that are used by these commands.

<br />

**Selecting input/output**

`-i` or `--input`

Accepts a string which is the path to a working directory or a specific data file. The path can be relative or absolute.

`-o` or `--output`

Accepts a string which defines the filename (no extension and no path) for your files. If AMiGA needs to merge results across multiple plates (due to the `--subset` or `--merge-summary` arguments) but no filename is passed here, AMiGA will instead give files a unique time stamp (e.g. 2020-08-26_09-09-59).

<br/>

**Reducing samples**: see [Data Subsetting](/amiga/doc/subsetting.html) for more details.

`-f` or `--flag`

Defines which wells should be excluded form the analysis. The proper syntax for the argument is to define the Plate_ID followed by a colon (`:`) followed by  Well IDs separated by commmas (`,`). User can concatenate flags for different plates with a semicolon (`;`). For example, `--flag "ER1_PM1-1:A11,C2;ER1_PM1-2:B3,B4,D5"`. 

`-s` or `--subset`

Defines which files to be included in the analysis. The proper syntax for the argument is to define the variable of interest (e.g. `Isolate`) followed by a colon (`:`) followed by the values of the variable of interest separated by commmas (`,`). User can susbet on different variables by concatenatig arguments with a semicolon (`;`). For example, `--subset Isolate:E.coli,C.diff;Antibiotics:None,Clindamycin`.

<br/>

**Selecting time points**

`-t` or `--interval`

Used to define the interval between time points in the data passed to `AMiGA`. Users can pass a single integer (e.g. `--interval 900`) or specify the interval for each plate (e.g. `--interavl "ER1_PM1-1:600;ER1_PM1-2:900;ER1_PM2-1:450"`). For the latter option, users must pass the `Plate_ID` followed by a colon (`:`) followed by the interval. User can set the interval for multiple plates by concatenating arguments with a semicolon (`;`) The default is 600 seconds. The default can be also modified in the `libs/config.py` file. In addition, user can modify the unit of time in the data input and output.

`-sfn` or `--skip-first-n`

Accepts an integer which is the number of time points to ignore in the beginnning. Default is 0. 

`-tss` or `--time-step-size`

Accepts an integer which is used to define how many time points to include in the GP model. For example, 1 indicates that each time point is included (i.e. T0, T1, T2, ...); 2 indicates that every other time point is included beginning with the first one (i.e. T0, T2, T4, ...); and so on. The default value is 1. Thinning of the data helps speed up the GP inference and reduction of the input to the GP model does not drastically alter overall growth curve shape as long as the input growth curve is smooth and does not exhibit acute spikes or dips. However, thinning of the data may alter estimates of growth parameters that describe fast dynamics (e.g. lag time and adaptation time) so please verify that your analysis would not be affected by reducing time points before doing so. 

`--keep-missing-time-points`

A boolean argument which forces `AMiGA` to model time points at which some replicates may have missing values. This is only applicable if user requests pooling with `--pool-by` argument. For example, you are pooling two plates where OD was measured hourly for a total of 25 time points for the first plate and 27 time points for the second plates. By default, AMiGA willd drop the last time points because the number of observations at that time is lower than the number of input growth curves. If you keep missing time points, `AMiGA` will include these less-sampled time points in the GP model.

<br/>

**Hypothesis testing**: see [Test Hypotheses](/amiga/doc/hypothesistesting.html) for more details.

`-y` or `--hypothesis`

Defines the hypothesis for GP regression model to be tested by AMiGA on the user-passed data. See [Test Hypotheses](/amiga/doc/hypothesistesting.html) for more details.

`-np` or `--number-permutations`

Accepts an integer and defines how many permutations to perform for estimating the null distribution of the Bayes Factor.  for more details. The default value is 20.

`-fdr` or `--false-discovery-rate`

Accepts an integer and defines the False Discovery Rate (FDR) threshold used in hypothesis testing. The default value is 10.

`--subtract-control`

A boolean argument on whether to condition samples on controls, i.e., subtract control growth curve(s) from treatment growth curve(s), before testing for differences in growth using GP inference.

<br/>

**Pooling and normalization**: see [Pool Replicates](/amiga/doc/hypothesistesting.html) or [Normalize Parameters](/amiga/doc/normalizing.html) for more details.

`--pool-by`

Accepts comma-separated list of variables in mapping files that will be used to specify how samples will be grouped. 

`--group-by`

Accepts comma-separated list of variables in mapping files that will be used to specify which wells are control samples. Only used for analysis with pooled replicates.

`--normalize-by`

Accepts comma-separated list of variables in mapping files that will be used to specify which wells are control samples. Only used for analysis with pooled replicates.

<br/>

**Plotting**

`--plot`

A boolean argument. If invoked, `AMiGA` will plot 96-well plates and overlay actual and model-predicted growth curves. If invoked with `amiga.py fit`, yellow dashed lines indicate the GP model prediction. 

`--plot-derivative`

A boolean argument that would save grid plots for 96-well plates of the estimated rate of change of OD in each well. Values above zero correspond to positive rate of change (i.e. growth or increase in OD) and values below zero correspond to negative rate of change (i.e. death or reduction in OD). See [Summarizing & Plotting Data](/amiga/doc/plotting.html) for more details on the colors in plots. In the plot, yellow dashed lines indicate the GP model prediction. 

`--dont-plot`

If you would like to run a basic summary or growth fitting (i.e. **\-\-only-basic-summary**) of the data (no GP modelling) without plotting figures, you should pass this argument.

`--dont-plot-delta-od`

If you woould lik to test hypotheses and plot only the growth curves without the functional OD difference between two growth curves. See [Hypothesis Testing](/amiga/doc/hypothesistesting.html) for more details.  

<br/>

**saving input/output**

`--merge-summary`

Instead of plate-specific summary files, a single summary file will be reported in the `summary` folder. This will occur by default if user requests any subsetting of the data.

`--save-cleaned-data`

A boolean argument that will save the data OD extracted from the files provided in the `data` folder. These derived data files will not include any meta-data and will be formatted as time x samples and saved in the `derived` folder. Column headers corresponds to the `Well` column in the corresponding key files saved in `summary` folder.

`--save-gp-data`

The result of this argument depends on whether a user requested pooling.

<ins>Without pooling</ins>: `--save-gp-data` that will save the different variants of the growth curve OD data before and after it has been fit with Gaussian Process model. Data will be formatted in a long-format where each row is a specific time point for a specific well in a specific plate. Identifying columns include `Time`,`Sample_ID`,`Plate_ID`. The `Plate_ID` corresponds to the name of the file from which the growth curve was processed. The `Sample_ID` corresponds to a specific well, see corresponding key files saved in `summary` for which well. Data columns include the following:
* `Time`: time points. 
* `GP_Input`: OD after log-transformation and baseline (OD at T0) subtraction. This is the OD variant explicitly modeled by a GP. OD at all time points will be included here even if user requested thinning of the data during modelling. 
* `GP_Output`: Predicted OD by the GP model based on the thinned `GP_Input`.
* `GP_Derivative`: Predicted derivative of the OD (dOD/dt) by the GP model.
* `OD_Growth_Data`: `GP_Input` that has been converted back to real OD but still baseline-corrected.
* `OD_Growth_Fit`: `GP_Output` that has ben converted back real OD but still baseline-corrected.
* `OD_Fit`: Predicted fit for the original OD. This is essentially `OD_Growth_Fit` but with baseline (OD at T0) added back.

<ins>With pooling</ins>: `--save-gp-data` will save a table in a long format where each row is a time point for each unique condition and columns include:
*  Meta-data variables that were passed to the `--pool-by` argument
* `Time`
* `mu`: mean of th predicted OD
* `mu1`: the mean of the predicted derivative of OD
* `Sigma`: covariance of the predicted OD
* `Sigma1`: covariance of the predicted derivative of OD
* `Noise`: the measurement noise. If `amiga fit` was run with the `--fix-noise` agument, the value would be the sum of the model estimated measurement noise (which should be negligible) and empirically estimated measurement noise. Otherwise, the value would be the model estimated measurement noise and should be the same value for all time points in each condition. 

`--save-mapping-tables`

A boolean argument that will ask force AMiGA to save all internally-generated or -parsed mapping files into `mapping` folder.

<br/>

**Technical adjustments to GP modelling**


`--sample-posterior`

A boolean argument on whether to infer summary statistics for all growth parameters. If not called, `AMiGA` povides a single estimate of each growth parameter based on the mean latent function of the GP model. If called, `AMiGA` samples from the posteiror distribution (`n` times, where `n` is defeined in the `libs/config.py` file), compute the growth parameter for each sampled curve, then returns the mean and the standard deivation of the mean of each growth parameter. See [Pool replicates](/amiga/doc/pooling.html) for more details.

`--fix-noise`

A boolean argument that dictates whether timepoint-specific noise should be included in the GP model. A GP model inherently models noise (error) as a single hyperparameter because it assumes that the noise is the same across all time points. This may not be the case for all growth data. Here, uers can opt for `AMiGA` to estimate the variance of the data (in pooled growth fitting) and pass those estimates of noise to the GP model which will fix the noise at each time point according to the pre-computed variance. Results seldom vary between standad GP model and fixed-noise GP models. However, the latter is more prone to overfitting and is computationally slower. See [Pool replicates](/amiga/doc/pooling.html) for more details.
 
`--include-gaussian-noise`

A boolean argument on whether plotted credible intervals should include estimated Gaussian noise in addition to the credible interval of the posterior of the latent function. This currently is only applicable for hypothesis testing plots. If called in conjunction with `--fix-noise` the credible interval will likely vary over time. 

<br />

**User communication**

`-v` or `--verbose`

A boolean argument. If invoked, `AMiGA` will communicate via the command terminal with the user by sharing in real time more details about the input, data processing, and results. This is extremely helpful for troubleshooting. 

<br/>
