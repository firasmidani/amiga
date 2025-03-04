---
layout: page
title: "Compare Parameters"
category: doc
date: 2020-11-05 15:26:50
use_math: false
order: 11
---
<!-- AMiGA is covered under the GPL-3 license -->
**Table of Contents**

* TOC
{:toc}

<br />

#### Basic usage

Users can directly compare growth parameters for two samples or two conditions using `compare.py`. This assumes that you have already analyzed your samples by pooling with `AMiGA`. See [How to infer summary statistics for pooled replicates?](/amiga/doc/pooling.html#how-to-infer-summary-statistics-for-pooled-replicates?). For example, let's say you ran the following command:

```bash
amiga fit -i /Users/firasmidani/experiment/ -o CD2015 --pool-by "Substrate,Isolate" --sample-posterior 
````

The `--sample-posterior` argument asks `AMiGA` to compute summary statistics for growth parameters (i.e., mean and standard deviations). Next, the following command will compare the growth of the CD2015 isolate in PM 1 on fructose and trehalose. It will generate the following table. 

```bash
amiga compare -i /Users/firasmidani/experiment/summary/CD2015_summary.txt -o CD2015_Fructose_vs_Trehalose -s "Substrate:D-Fructose,D-Trehalose;Isolate:CD2015;PM:1" --confidence 95
```
- `-i` must point to the summary file genearted by AMiGA.
- `-o` assigns a filename for the results, otherwise, the filename will be a unique time stamp.
- `-s` susbetting arguments must reduce analysis to only two conditions, otherwise, this command will fail and result in an error.
- `--confidence` allows you to change the magnitude of the confidence interval. The default is 95. 

<br/>

#### Typical output

This will generate the below which will be saved in the same directory as the input file. Keep in mind that the below example is based only on two technical replicates for each condition, so the statistical power is pretty low, but the differences in growth dynamics are pretty clear. See Figure 2 of AMiGA manuscript for growth curves. 

<br/>

| Substrate                 | D-Trehalose | D-Fructose | D-Trehalose       | D-Fructose        |            |
| ------------------------- | ----------- | ---------- | ----------------- | ----------------- | ---------- |
| PM                        | 1           | 1          | 1                 | 1                 |            |
| Isolate                   | CD2015      | CD2015     | CD2015            | CD2015            |            |
| Parameter                 | Mean        | Mean       | 95.0% CI          | 95.0% CI          | Sig. Diff. |
| AUC (log)                 | 36.435      | 36.218     | [35.933,36.938]   | [35.888,36.547]   | FALSE      |
| Death (log)               | 0.095       | 1.427      | [-0.037,0.226]    | [1.335,1.519]     | TRUE       |
| Diauxie                   | 0           | 0          | NA                | NA                | FALSE      |
| Death Rate                | -0.041      | -0.307     | [-0.106,0.023]    | [-0.344,-0.270]   | TRUE       |
| Growth Rate               | 0.403       | 0.604      | [0.331,0.476]     | [0.571,0.637]     | TRUE       |
| Carrying Capacity (log)   | 2.125       | 2.259      | [2.076,2.173]     | [2.219,2.298]     | TRUE       |
| Lag Time                  | 0.474       | 1.037      | [0.187,0.761]     | [0.929,1.145]     | TRUE       |
| Adaptation Time           | 0.167       | 0.475      | [0.167,0.167]     | [0.195,0.755]     | TRUE       |
| Time at Max. Death Rate   | 20.817      | 19.723     | [18.147,23.486]   | [18.831,20.615]   | FALSE      |
| Time at Max. Growth Rate  | 0.187       | 2.62       | [0.005,0.368]     | [2.417,2.823]     | TRUE       |
| Time at Carrying Capacity | 17.492      | 9.967      | [13.882,21.102]   | [9.408,10.526]    | TRUE       |
| Doubling Time             | 1.732       | 1.148      | [1.419,2.046]     | [1.086,1.211]     | TRUE       |


<br/>
**Comparing growth curves analyzed in different summary files**

Let's say you want to compare two samples but their growth summary are in different files. You can manually create a new summary file with only the two samples (rows) that you are interested in anlayzing, then passing this file to the `-i` argument. But you can also pass multiple `-i` arguments to `AMiGA`. The below will find the growth summary for CD2015 and CD1007 on fructose the compare them against each other.

```bash
amiga compare -i /Users/firasmidani/experiment/summary/CD2015_summary.txt -i /Users/firasmidani/experiment/summary/CD1007_summary.txt -o CD2015_vs_CD1007_on_Fructose -s 'Substrate:D-Fructose;Isolate:CD2015,CD1007' --confidence 95
```

<br />
#### Command-line arguments

To see the full list of arguments that `amiga compare` will accept, run

```bash
amiga compare --help
```
which will return the following message

```bash
usage: amiga [-h] -i INPUT -o OUTPUT -s SUBSET [--confidence CONFIDENCE]
             [--verbose]

Compare two growth curves

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
  -o OUTPUT, --output OUTPUT
                        ouptut filename including path
  -s SUBSET, --subset SUBSET
  --confidence CONFIDENCE
                        Must be between 80 and 100. Default is 95.
  --verbose
```

<br/>
See more details for these arguments in [Command Line Interface](/amiga/doc/command-line-interface.html)
