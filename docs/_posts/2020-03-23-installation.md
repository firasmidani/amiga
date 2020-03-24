---
layout: page
title: "Installation"
category: doc
date: 2020-03-23 16:39:13
---

#### [Required] Download repository or code

You can do either of the following
<br />
- Clone via the command terminal: `git clone https://github.com/firasmidani/amiga.github`
- Download manually: go to <a href="https://github.com/firasmidani/github">AMiGA</a>, click the green button <span style="color:#ffffff;background-color:#2ab748">&nbsp;Clone or download&nbsp;</span> on top right corner, then click <span style="color:#075bd0;">Download ZIP</span>.

Please extract or download the ZIP folder in a location that you can easily access. You will have to point `Python` to this folder whenever you want to run `AMiGA`.
<br /><br />

#### [Required] `Python`

* If you are a Max or Unix user, your machine will have `Python` pre-installed.
* If you are a Windows user and have not previously worked with `Python`, I recommend a `Python` distribution such as `Anaconda`. See this useful [guide](https://fangohr.github.io/blog/installation-of-python-spyder-numpy-sympy-scipy-pytest-matplotlib-via-anaconda.html) on installing Python. If you run into

`AMIGA` was written in `Python 3` and should be compatible with `Python>=2.7`.
<br /><br />

#### [Optional] Set-up a local `Python` environment

I highly recommend that you set-up a virtual `Python` environment for running `AMiGA` whether you are a new or experienced user of `Python`.  A virtual environment is a self-contained directory that will contain a copy of your `Python` installation, plus a limited number of additional packages that you select. You can create a virtual environment customized for `AMiGA` with the only packages that it needs. The main advantages are (1) you will know that your environment meets the requirements of `AMiGA` and (2) that it would not contradict the requirements of other programs that use your main `Python` installation.

Please follow these guides for creating virtual environments: a <a href="https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/">general guide</a> or a <a href="https://uoa-eresearch.github.io/eresearch-cookbook/recipe/2014/11/20/conda/">Anaconda-specific guide</a>. You can also try out my simple but limited instructions below. If the instructions below are not working for you, simply go to the before-mentioned guides and follow their more detailed instructions.

**Check your Python installation and version**

* Your machine may have multiple installations of `Python`. You must use the same installation every time you run AMiGA.

    - You can find out which Python installation you are calling with the following commands.

    `which python` in MacOS or Unix
    `where python` in Windows

    - You can call a specific Python installation by pointing to its full path.

    `/Users/firasmidani/python3`  for MacOS or Unix
    `C:\Users\firasmidani\python3` for Windows

2. Make sure that you are using `Python 3`. You can out find out the version of your `Python` installation

    `python --version`

**MacOS and Unix users of Python 3**


<br /><br />


#### [Required] Python package requirements

Please follow these instructions to install all requirements. If you plan to run `AMiGA` in a virtual environment, you need to activate the environment first (e.g. source /Users/firasmidani/example/amiga-python-environment).

Change your directory toAMiGA


If you are using native installation of `Python`:

```pip install -r requirements.txt```




```bash
cd /Users/firasmidani/amiga/
```
