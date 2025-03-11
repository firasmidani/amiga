---
layout: page
title: "Installing AMiGA with miniforge"
category: doc
date: 2025-03-11 11:07:00 AM
use_math: true
---

**Table of Contents**

* TOC
{:toc}
<br />

### Install Miniforge3

Install `Miniforge` from [here](https://github.com/conda-forge/miniforge) and follow the steps that are specific to your operating system.

Miniforge will allow you to use `conda` and `mamba` which are software that can help you install and run `AMiGA` on your machine. These software are environment management systems. In other words, they will let you create a unique environment on your machine to run `AMiGA` without affecting other environments that you create to run other software.

`mamba` is essentially a faster alternative to conda. We will use it below. If you prefer to use `conda`, then, you can simply replace all `mamba` instances below with `conda`.

### Initialize mamba/conda

Open your `Terminal` and type

```bash
mamba init
```

This will simply tell your computer where to find `mamba` on your machine. Close the terminal.

### Install AMiGA

Re-open the terminal and type

```bash
mamba create -n amiga-env bioconda::amiga
```

This will create an environment called `amiga-env` and install `AMiGA` inside this environment. This step will take a couple of minutes.

### Test AMiGA

```bash
mamba activate amiga-env
```

This will activate the `amiga-env` environment. Now, your terminal should know where and how to run `AMiGA`. So, try to pull up the help menu. 

```bash
amiga -v              # display AMiGA version numbers
amiga -h              # help menu
amiga summarize -h    # help menu for summarize command
```

You should get the following

```bash
usage: amiga <command> [<args>]

The most commonly used amiga commands are:
    summarize       Perform basic summary and plot curves
    fit             Fit growth curves
    normalize       Normalize growth parameters of fitted curves
    compare         Compare summary statistics for two growth curves
    test            Test a specific hypothesis
    heatmap         Plot a heatmap
    get_confidence  Compute confidence intervals for parameters or curves
    get_time        Get time at which growth reaches a certain value
    print_defaults  Shows the default values stored in libs/config.py

See `amiga <command> --help` for information on a specific command.
For full documentation, see https://firasmidani.github.io/amiga

positional arguments:
  command        Subcommand to run. See amiga --help for more details.

options:
  -h, --help     show this help message and exit
  -v, --version  Show version and exit.
```

### Run AMiGA in the future

Open your `Terminal`, type `mamba activate amiga-env`, and you are ready to analyze your data. 
