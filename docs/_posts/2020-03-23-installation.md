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

`AMiGA` is designed for use by scientists with different backgrounds in bioinformatics. To make `AMiGA` accessible for users with no experience in using `Python` or a command terminal, here I have added detailed instructions on how to install `AMiGA` under different computing conditions.

<br />

**Note for begginers:** AMiGA is written in the Python programming language, which is simply a language that allows a programmer to communicate with a computer and dictate how certain tasks are performed. Once you write code in Python, you can execute it (i.e. run it) using the command terminal. A command terminal is simply an interface through which you can submit commands to a computer. These commands are not limited to Python. So, to run AMiGA, you need the code for AMiGA, Python installed in your computer, and several Python-specific packages. What is a package? AMiGA is a packge or in other words a collection of scripts written in Python. AMiGA however also uses packages written by others such as Pandas, Scipy, and Numpy. These are very popular packages for scientific computing. The below instructions explain step-by-step how to set-up AMiGA on your computer.  

<br /> 

#### 1. Download repository or code [Required] 

You can do either of the following
<br />
- Clone via the command terminal: `git clone https://github.com/firasmidani/amiga.git`
- Download manually: go to <a href="https://github.com/firasmidani/amiga">AMiGA</a>, click the green button <span style="color:#ffffff;background-color:#2ab748">&nbsp;Code&nbsp;</span> on top right corner, then click **Download ZIP**.

Please extract or download the ZIP folder in a location that you can easily access. You will have to point `Python` to this folder whenever you want to run `AMiGA`. You can altenatively create an alias that always points to the `amiga.py` file. See [Terminal Alias](#terminal-alias-[Optional])
<br /><br />

#### 2. Install Python [Required] 

`Python` is a programming language commonly used for scientific computing. If you are a Max or Unix user, your machine should have `Python` pre-installed. If you are new to `Python` or use Windows, I recommend a `Python` distribution such as `Miniconda3` (**AMiGA is  only compatible with Miniconda3-4.5.4 or lower**. Go [here](https://repo.anaconda.com/miniconda/) to download this version.) `Miniconda3` is available for all operating systems (Windows, Linux, and Mac).

`AMIGA` was written in `Python 3` and should be compatible with `Python>=2.7`. See this useful [guide](https://fangohr.github.io/blog/installation-of-python-spyder-numpy-sympy-scipy-pytest-matplotlib-via-anaconda.html) on installing Python.
<br /><br />

**Check your `Python` installation and its version**

Your machine may have multiple installations of `Python`. You should use the same installation every time you run `AMiGA`. So, it helps to know the following:

1. You can find out which Python installation you are calling with the following commands.

    `which python` in MacOS or Unix

    `where python` in Windows

2. If this is not the installation that you need, you can call a specific Python installation by pointing to its full path.

    `/Users/firasmidani/python3`  for MacOS or Unix

    `C:\Users\firasmidani\python3` for Windows

3. Make sure that you are using `Python 3`. You can identify the version of your `Python` installation with:

    `python --version`
<br /><br />

#### 3. Set-up a Python virtual environment [Optional] 

I highly recommend that you set-up a virtual `Python` environment for running `AMiGA` whether you are a new or experienced user of `Python`.  A virtual environment is a self-contained directory that will contain a copy of your `Python` installation, plus a limited number of additional packages that you select. You can create a virtual environment customized for `AMiGA` with the only packages that it needs. The main advantages are (1) you will know that your environment meets the requirements of `AMiGA` and (2) that it would not contradict the requirements of other programs that use your main `Python` installation.

Please follow these guides for creating virtual environments: a <a href="https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/">general guide</a> or a <a href="https://uoa-eresearch.github.io/eresearch-cookbook/recipe/2014/11/20/conda/">conda-specific guide</a>. You can also try out my simple but limited instructions below. If the instructions below are not working for you, simply go to the before-mentioned guides and follow their more detailed instructions.
<br /><br />

**Mac OS and Unix users**

1. If you are using `Python 3.3` or newer, `venv` is a standard package and requires no additional installation. Otherwise, you can install `virtualenv`.

    `python -m pip install --user virtualenv`

2. Set-up the environment. Here, I name it `virtual_environment` and store it in my home directory.

    `python -m virtualenv /Users/firasmidani/virtual_environment`  you can substitute `venv` for `virtualenv`

3. Activate the environment. You will need to do this every time you are working with `AMiGA`.

    `source /Users/firasmidani/virtual_environment/bin/activate`

<br />
**Windows Users**

1. If you are using `Python 3.3` or newer, `venv` is a standard package and requires no additional installation. Otherwise, you can install `virtualenv`.

    `python -m pip install --user virtualenv`

2. Set-up the environment. Here, I name it `virtual_environment` and store it in my home directory.

    `python -m virtualenv C:\Users\firasmidani\virtual_environment`  you can substitute `venv` for `virtualenv`

3. Activate the environment. You will need to do this every time you are working with `AMiGA`.

    `C:\Users\firasmidani\virtual_environment\Scripts\activate`

<br />
**Anaconda or Miniconda users**

1. Open your Anaconda Prompt or Miniconda Prompt terminal.

2. Set-up the environment. Here, I name it `virtual_environment`.

    `conda create -n virtual_environment`

3. Activate the environment. You will need to do this every time you are working with `AMiGA`.

    `conda activate virtual_environment`

<br />

#### 4. Install Python package requirements [Required] 

Please follow these instructions to install all requirements. See `amiga/requirements.txt` for a full list of dependencies.

- If you plan to run `AMiGA` in a virtual environment, you need to activate the environment first.

    `source /Users/firasmidani/virtual_environment/bin/activate`  for MacOS or Unix users

    `C:\Users\firasmidani\virtual_environment\Scripts\activate` for Windows


- Change your directory to `AMiGA` which you cloned or downloaded from Github.

    `cd /Users/firasmidani/amiga` for MacOS or Unix users

    `cd "C:\Users\firasmidani\amiga"` for Windows

- If you are using Anaconda or Miniconda, you can install the requirements with the following

    `conda config --add channels conda-forge`

    `conda install --file requirements.txt`

- Otherwise, you can use pip as follows:

    `pip install -r requirements.txt`

<br />

**Troubleshooting**

- If you are using `Anaconda` or `Miniconda` and installation of specific package(s) fails or results in an error, you can try:

    `conda config --add channels conda-forge`

    `conda install -c conda-forge PACKAGE_NAME`

- If you are installing requirements with `Anaconda` or `Miniconda` and you are running into an error or getting stuck at `Solving environment: failed with initial frozen solve`. You may need to downgrade `conda` and its `python` version. **AMiGA is  only compatible with Miniconda3-4.5.4 or lower**. Go [here](https://repo.anaconda.com/miniconda/) to download this version. Try the following (you may also need to re-install conda before this step). Then, you may have to re-create your virtual environemnt before re-attempting to install the requirements.

    `conda config --set auto_update_conda True`

    `conda config --set allow_conda_downgrades True`

    `conda install python=3.6`
    
    `conda install conda=4.6`



<br />

#### 5. Set-up Terminal alias [Optional] 

These instructions only apply to for MacOS and Unix users and are not applicable for Windows or conda users. 

Instead of pointing the terminal to the `AMiGA` code for every command, we can create a short alias to be used in the command terminal. You can create an alias by adding the following command to the `~/.zshrc` file if you are using `zsh` terminal or to `~/.bash_profile` if you are using the `bash` terminal. If you are unsure which one you are using, try `echo $SHELL` in you terminal. Make sure to edit the path to match the location of your `amiga.py` file. 

```alias amiga="python /Users/firasmidani/rab_fm/git/amiga/amiga.py"```

These environmental variables files are executed by your machine upon login. Because you are already logged-in, you will need to execute them again before the alias takes effect. 

`source ~/.zshrc` o `source ~/.bash_profile`

The alias essentially associates the word `amiga` with running `python` on the `amiga.py` file in the `AMiGA` directory. So now instead of calling the longer command in the terminal

```python /Users/firasmidani/rab_fm/git/amiga/amiga.py --help```

You can simply call

`amiga`

which would accordingly print the help message for `AMiGA`.

```bash
usage: amiga.py <command> [<args>]

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
