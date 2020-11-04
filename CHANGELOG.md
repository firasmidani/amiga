# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

## [1.1.0] - 2020-11-04

New feature:
- `compare.py`: allows users to specifically test for differences in growth parameters. It applies the same logic used by `HypothesisTest()` for generating `*_params.txt`.

Regarding hypothesis Testing: 
- To simplify code and avoid ambiguity for users, hypothesis testing is now limited to testing for differences due to a single binary condition. Differential testing for effects due to multiple and/or non-binary variables are no longer supported. 
- Posterior sampling to generate summary statistics for parameter estimates are now allowed. This however assumes that `ARD` (i.e. automatic relevance determination) is always `True`, which is currently the case. Users cannot change `ARD` preference for kernels in `libs/config.py`. 
- `computeFullDifference()` is now always called as a class method. 
- Functional differences is always computed and reported by `HypohesisTest()` in the long-form `*_report.txt` and short-form `*_log.txt` files.
- Functional differences is now limited to a single summary metric based on on all time points and confidence intervals are now immediately generated based on 100 permutations. 
- Functional differences are also exported in `*_func_diff.txt` which allows users to easily plot the data. 
- `—plot-delta-od` is now True by default. This seems reasonable now that testing is limited to single binary variables.

Regarding growth parameters:
- Major modifications to `libs/params.py` (and `libs/config.py`) that allow users to select which growth parameters are reported by `AMiGA`. 
- `x_k`, `x_gr`, `x_dr` are now referred to as `t_k`, `t_gr`, `t_dr` respectively. This naming scheme is more intuitive. 
- Two new functions `prettifyParameterReport()` and `articulateParameters()` are added. These functions are needed for generating human-readable reports for differential testing of growth parameters, whether by `HypothesisTest()` or `compare.py`.
- Death was previously computed as the fraction of total growth (OD) lost after reaching carrying capacity. Now, it is computed as the total OD lost after reaching carrying capacity. 
- Parameter defaults for diauxie detection have been updated. Internally, parameter defaults are now called only by `libs/diauxie.py`. 


## [1.0.0] - 2020-08-26

Initial commit for [AMiGA](https://firasmidani.github.io/amiga) which is a major update to [Phenotypic-Characterization](https://firasmidani.github.io/phenotypic-characterization).
