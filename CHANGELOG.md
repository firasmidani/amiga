# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

## [3.0.0] - 2025-03-03

- AMiGA was refactored into a Python package (no longer as standalone scripts).
- AMiGA was updated to work with Python versions <= 3.12
- AMiGA was upoaded to PyPi and Anaconda, and can now be installed with pip or conda.

## [2.0.0] - 2021-04-21

New interface: 
- As AMiGA got bloated, it became more confusing for users. I have re-structured the command-line interface. AMiGA now has one main command `amiga.py` and sub-commands including `Summarize`, `Fit`, `Normalize`, `Compare`, `Heatmap`, `Test`, `Get-Confidence`, and `Get-Time`. Most of these sub-commands were already implemented in previous verisons but were called as arguments (e.g., `--only-basic-summary`).

New features: 
- `Heatmap` has a lot more features including exporting data, adjusting aesthetics, labelling rows or columns by unique colors, and highlighting individual labels with a unique fontstyle. Users can now alter how heatmaps handle missing data using command-line arguments. 
- Normalization of growth parameters is now a separate command: `Normalize`. Users can now normalize either by subtraction or division. 
- Users can now explicitly request confidence intervals either for growth parameters or growth curves using the `Get-Confidence` section.
- Growth curves are often used for identifying the time it takes for OD to reach a certain threshold. Users can now do this with `Get-Time`. 
- Users can now subtract background OD, control samples, and select their preferred approach for handling non-positive values.

Improved and expanded documentation on how to use the many features of `AMiGA` and nuances on how to interpret results.


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

Initial commit for [AMiGA](https://github.com/firasmidani/amiga) which is a major update to [Phenotypic-Characterization](https://github.com/firasmidani/phenotypic-characterization).
