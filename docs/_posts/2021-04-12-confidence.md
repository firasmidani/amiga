---
layout: page
title: "Estimate Confidence"
category: doc
date: 2021-04-12 13:00:05
use_math: false
order: 12
---

<!-- AMiGA is covered under the GPL-3 license -->

The `get_confidence` function allows you to easily estimate the confidence intervals for either growth parameters or the predicted growth curves.

<br />
#### Estimating confidence intervals for growth parameters

<br />
`AMiGA` can estimate mean and standard deviation of growth parameters if it fit growth curves based on multiple replicates. See [How to infer summary statistics for pooled replicates](/amiga/doc/pooling.html). For example, 

<br />

```bash
python amiga.py fit -i /Users/firasmidani/experiment/ -o "pooled_analysis" --pool-by "Isolate,Substrate" --sample-posterior 
```

<br />

The above command will generate a summary file `summary/pooled_analysis_summary.txt` that will include the estiamted mean and standard deviation for a variety of growth parameters. If you would like to estimate the confidence intervals for these parameters, you can do the following

<br />

```bash
python amiga.py get_confidence -i /Users/firasmidani/experiment/summary/summary/pooled_analysis_summary.txt --type 'Parameters' --confidence 95
```

<br />

This will generate a new file `summary/pooled_analysis_summary_confidence.txt` where it will include also the lower and upper bounds for the 95% confidence interval of all growth parameters.

<br />

#### Estimating confidence intervals for growth curves

`AMiGA` can pool replicate curve and model them jointly. If requested by the user (`--save-gp-data`, it will save the predicted mean and covariance for the growth curves as well as the estimated Gaussian noise. For example, 

<br />

```bash
python amiga.py fit -i /Users/firasmidani/experiment/ -o "pooled_analysis" --pool-by "Isolate,Substrate" --sample-posterior --save-gp-data
```

<br />

The above command will generate a text fiel `derived/pooled_analysis_gp_data.txt` which will have columns for the sample's meta-data, in addition to:
- `mu`: mean of the growth function per time point
- `Sigma`: variance of the growth function per time point
- `mu1`: mean of the growth rate function per time point
- `Sigma1`: variance of the growth function per time point
- `Noise`: measurement Noise (time-independent by default but time-dependent if you also use the `--fix-noise` argument).

<br />

You can estimate the confidence intervals for the growth function and growth rate function as follows:

<br />

```bash
python amiga.py get_confidence -i /Users/firasmidani/experiment/derived/pooled_analysis_gp_data.txt --type 'Curves' --confidence 95
```

<br />

This will generate a new file `derived/pooled_analysis_gp_data_confidence.txt` where it will include also the lower and upper bounds for the 95% confidence interval of all growth parameters. This copy of the input file will have four additional columns for the lower (`Low`) and upper (`Upper`) confidence intervals of the growth function, and the lower (`Low1`) and upper (`Upper1`) confidence intervals of the growth rate function

By default, `get_confidence` will compute the confidence intervals without including sampling uncertainty (i.e. measurement noise). If you would like to include noise in the confidence interval, you must pass `--include-noise`. 

<br />

#### Command-line arguments

To see the full list of arguments that `amiga compare` will accept, run

```bash
python compare.py --help
```
which will return the following message

```bash
usage: amiga.py [-h] -i INPUT --type {Parameters,Curves}
                [--confidence CONFIDENCE] [--include-noise] [--over-write]
                [--verbose]

Compute confidence intervals for parameters or curves.

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
  --type {Parameters,Curves}
  --confidence CONFIDENCE
                        Must be between 80 and 100. Default is 95.
  --include-noise       Include the estimated measurement noise when computing
                        confidence interval (For Curves Only).
  --over-write          Over-write file otherwise a new copy is made with
                        "_confidence" suffix
  --verbose
```

<br/>
See more details for these arguments in [Command Line Interface](/amiga/doc/command-line-interface.html)
