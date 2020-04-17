---
layout: page
title: "Command Line Interface"
category: doc
date: 2020-04-16 21:42:08
order: 10
use_math: false
---

To see the full list of arguments that `AMiGA` will accept, you can pass it the `-h` or `--help` argument.

```bash
python main.py --help
```

which will reveal the followin g message

```bash
usage: main.py [-h] -i INPUT [-f FLAG] [-s SUBSET] [-y HYPOTHESIS]
               [-t INTERVAL] [-p] [-v] [-np NUMBER_PERMUTATIONS]
               [-nt TIME_POINTS_SKIPS] [-fdr FALSE_DISCOVERY_RATE]
               [--merge-summary] [--plot-derivative] [--only-basic-summary]
               [--save-all-data] [--save-derived-data] [--save-fitted-data]
               [--save-transformed-data] [--only-print-defaults]
               [--perform-substrate-regression] [--dont-subtract-control]
               [-o OUTPUT]

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
  -f FLAG, --flag FLAG
  -s SUBSET, --subset SUBSET
  -y HYPOTHESIS, --hypothesis HYPOTHESIS
  -t INTERVAL, --interval INTERVAL
  -p, --plot
  -v, --verbose
  -np NUMBER_PERMUTATIONS, --number-permutations NUMBER_PERMUTATIONS
  -nt TIME_POINTS_SKIPS, --time-points-skips TIME_POINTS_SKIPS
  -fdr FALSE_DISCOVERY_RATE, --false-discovery-rate FALSE_DISCOVERY_RATE
  --merge-summary
  --plot-derivative
  --only-basic-summary
  --save-all-data
  --save-derived-data
  --save-fitted-data
  --save-transformed-data
  --only-print-defaults
  --perform-substrate-regression
  --dont-subtract-control
```

**-i** or **--input**

Accepts a string which is the path to a working directory or a specific data file. The path can be relative or absolute.

**-f or --flag**

Defines which wells should be excluded form the analysis. See [Data Subsetting](/amiga/doc/subsetting.html) for more details.

**-s** or **--subset**

Defines which files to be included in the analysis. See [Data Subsetting](/amiga/doc/subsetting.html) for more details.

**-h** or **--hypothesis**

Defines the hypothesis for GP regression model to be tested by AMiGA on the user-passed data. See [Hypothesis Testing](/amiga/doc/hypothesistesting.html) for more details.

**-t** or **--interval**

Used to define the interval between time points in the data passed to `AMiGA`. The default is 600. The default can be also modified in the `libs/config.py` file. In addition, user can modify the unit of time in the data input and output.

**-v** or **--verbose**

A boolean argument. If invoked, `AMiGA` will communicate via the command terminal with the user by sharing in real time more details about the input, data processing, and results.

**-np** or **--number-permutations**

Accepts an integer and defines how many permutations to perform for estimating the null distribution of the Bayes Factor. See [Hypothesis Testing](/amiga/doc/hypothesistesting.html) for more details. The default value is 20.

**-nt** or **--time-poins-skips**

Accepts an integer and used to define how many time points to include in hypothesis testing. 1 indicates that each time point is included; 2 indicates that every other time point is included beginning with the first one; and so on. The default value is 11.

**-fdr** or **--false-discovery-rate**

Accepts an integer and defines the False Discovery Rate (FDR) threshold used in hypothesis testing. The default value is 20. See [Hypothesis Testing](/amiga/doc/hypothesistesting.html) for more details.  

**--merge-summary**

Instead of plate-specific summary files, a single summary file will be reported in the `summary` folder.

**--plot-derivative**

A boolean argument that would save grid plots for 96-well plates of the estimated rate of change of OD in each well. Values above zero correspond to positive rate of change (i.e. growth or increase in OD) and values below zero correspond to negative rate of change (i.e. death or reduction in OD). See [Summarizing & Plotting Data](/amiga/doc/plotting.html) for more details on the colors in plots.

**--only-basic-summary**

See [Summarizing & Plotting Data](/amiga/doc/plotting.html) for more details.

**--save-all-data**

A booolean argument that invokes `--save-derived-data`, `--save-fitted-data`, and `--save-transformed-data`.

**--save-derived-data**

A boolean argument that will save the data OD extracted from the files provided in the `data` folder. These derived data files will not include any meta-data and will be formatted as time x samples and saved in the `derived` folder. Column headers corresponds to the `Well` column in the corresponding key files saved in `summary` folder.

**--save-fitted-data**

A boolean argument that will save the transformed OD data after it has been fit with Gaussian Process. Data will be formatted as time x wells and saved in the `derived` folder. Column headers corresponds to the `Sample_ID` column in the corresponding key files saved in `summary` folder.

**--save-transformed-data**

A boolean argument that will save the OD data after it has been natural logarithm-transformed and baseline corrected. Data will be formatted as time x samples and saved in the `derived` folder. Column headers corresponds to the `Sample_ID` column in the corresponding key files saved in `summary` folder.

**--only-print-defaults**

A boolean argument that will print to the command terminal the parameters defined in the `libs/config.py` file. You can replace these default parameters to suit your analysis.

**--dont-subtract-control**

A boolean argument on whether to condition samples on controls (i.e. subtract control growth curve from treatment growth curve) before testing for differences in growth using GP inference. See [Hypothesis Testing](/amiga/doc/hypothesistesting.html) for more details.
