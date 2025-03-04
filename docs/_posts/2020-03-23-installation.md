---
layout: page
title: "Install AMiGA"
category: doc
date: 2020-03-23 16:39:13
order: 1
use_math: true
---
<!-- AMiGA is covered under the GPL-3 license -->

**Table of Contents**

* TOC
{:toc}
<br />

`AMiGA` is designed for use by scientists with different backgrounds in bioinformatics. It is compatible with Python version 3.12 or lower.

<br />

## Instructions for Advanced Users

`AMiGA` is available on `PyPI` and `bioconda` and can be installed by any of the following. I recommend that you create a virutal environment then install `AMiGA``. 

```
conda create -n amiga-env bioconda::amiga
```

```
mamba create -n amiga-env bioconda::amiga
```

```
pip install amiga
```

You can test your installation by pulling up the help mneu.

```
mamba activate amiga
amiga -h
```

If you would like to build `AMiGA` locally, then

```
git clone https://github.com/firasmidani/amiga
cd amiga
python -m build .
python -m pip install dist/amiga*.whl
```

## Instructions for Beginners

#### 1. Install Miniforge3

Install `Miniforge`. Follow instructions that are specific to your operating system here: 
https://github.com/conda-forge/miniforge

Miniforge will allow you to use `conda` and `mamba` which are software that can help you install and run `AMiGA` on your machine. These software are environment management systems. In other words, they will let you create a unique environment on your machine to run `AMiGA` without affecting other environments that you create to run other software beyond `AMiGA`.

`mamba` is essentially a faster alternative to conda. We will use it below. If you prefer to use `conda`, then, you can simply replace all `mamba` instances below with `conda`.

### 2. Initialize mamba/conda

`mamba init`

This will simply tell your computer where to find `mamba` on your machine. 

### 3. Install AMiGA

`mamba create -n amiga-env bioconda::amiga`

This will create an environment claled `amiga-env` and install `AMiGA` inside this environemnt. This step will take a couple of minutes.

### 4. Test AMiGA

`mamba activate amiga-env`

This wiill activate the `amiga-env` environment. Now, your terminal should know where and how to run `AMiGA`. So, let's pull up the help menu. 

`amiga -h`

You should get the following

```bash
usage: amiga <command> [<args>]

The most commnly used amiga.py commands are:
    summarize       Perform basic summary and plot curves
    fit             Fit growth curves
    normalize       Normalize growth parameters of fitted curves
    compare         Compare summary statistics for two growth curves
    test            Test a specific hypothesis
    heatmap         Plot a heatmap
    get_time        Get time at which growth reaches a certain value
    print_defaults  Shows the default values stored in libs/config.py

See `amiga.py <command> --help` for information on a specific command.
For full documentation, see https://firasmidani.github.io/amiga

positional arguments:
  command     Subcommand to run. See amiga.py --help for more details.

optional arguments:
  -h, --help  show this help message and exit
```
