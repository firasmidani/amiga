---
layout: page
title: "Comparing Parameters"
category: doc
date: 2020-11-05 15:26:50
use_math: false
order: 13
---

Users can directly compare growth parameters for two samples or two conditions using `compare.py`.

To see the full list of arguments that `AMiGA` will accept, you can pass it the `-h` or `--help` argument.

```bash
python compare.py --help
```

which will reveal the following message

```bash
usage: compare.py [-h] -i INPUT -o OUTPUT [-s SUBSET]
                  [--confidence CONFIDENCE]

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
  -o OUTPUT, --output OUTPUT
  -s SUBSET, --subset SUBSET
                        See amiga.py --subset for formatting tips.
  --confidence CONFIDENCE
                        Must be between 80 and 100. Default is 95.
```

For example, the following command will compare the growth of the CD2015 isolate in PM 1 on fructose and trehalose. It will generate the following table. 

```bash
python $HOME/rab_fm/git/amiga/compare.py \
	-i /Users/firasmidani/experiment/summary/CD2015_summary.txt \
	-o CD2015_Fructose_vs_Trehalose \
	-s 'Substrate:D-Fructose,D-Trehalose;Isolate:CD2015;PM:1' \
	--confidence 95
```

| Substrate                 | D-Trehalose | D-Fructose | D-Trehalose       | D-Fructose        |            |
| ------------------------- | ----------- | ---------- | ----------------- | ----------------- | ---------- |
| PM                        | 1           | 1          | 1                 | 1                 |            |
| Isolate                   | CD2015      | CD2015     | CD2015            | CD2015            |            |
| Parameter                 | Mean        | Mean       | 95.0% CI          | 95.0% CI          | Sig. Diff. |
| AUC (lin)                 | 114.532     | 112.546    | [110.683,118.381] | [109.667,115.425] | FALSE      |
| AUC (log)                 | 36.435      | 36.218     | [35.933,36.938]   | [35.888,36.547]   | FALSE      |
| Death (lin)               | 0.745       | 7.272      | [-0.270,1.760]    | [6.847,7.697]     | TRUE       |
| Death (log)               | 0.095       | 1.427      | [-0.037,0.226]    | [1.335,1.519]     | TRUE       |
| Diauxie                   | 0           | 0          | NA                | NA                | FALSE      |
| Death Rate                | -0.041      | -0.307     | [-0.106,0.023]    | [-0.344,-0.270]   | TRUE       |
| Growth Rate               | 0.403       | 0.604      | [0.331,0.476]     | [0.571,0.637]     | TRUE       |
| Carrying Capacity (lin)   | 7.487       | 8.557      | [7.062,7.913]     | [8.169,8.944]     | TRUE       |
| Carrying Capacity (log)   | 2.125       | 2.259      | [2.076,2.173]     | [2.219,2.298]     | TRUE       |
| Lag Time                  | 0.474       | 1.037      | [0.187,0.761]     | [0.929,1.145]     | TRUE       |
| Adaptation Time           | 0.167       | 0.475      | [0.167,0.167]     | [0.195,0.755]     | TRUE       |
| Time at Max. Death Rate   | 20.817      | 19.723     | [18.147,23.486]   | [18.831,20.615]   | FALSE      |
| Time at Max. Growth Rate  | 0.187       | 2.62       | [0.005,0.368]     | [2.417,2.823]     | TRUE       |
| Time at Carrying Capacity | 17.492      | 9.967      | [13.882,21.102]   | [9.408,10.526]    | TRUE       |
| Doubling Time             | 1.732       | 1.148      | [1.419,2.046]     | [1.086,1.211]     | TRUE       |

The above example is based only on two technical replicates condition, so the statistical power is pretty low, but the differences in growth dynamics are pretty clear. See Figure 2 of AMiGA manuscript for growth curves. 
