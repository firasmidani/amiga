---
layout: page
title: "Installing AMiGA with pip"
category: doc
date: 2025-03-11 11:07:00 AM
use_math: true
---

**Table of Contents**

* TOC
{:toc}
<br />

### Install Python
  
Download Python 3.12.9 from https://www.python.org/downloads/

Open the downloaded file. 

During installation, make sure to check a box that says "Add Python to PATH". 
<br>
<br>

### Verify installation

Open your `Command Prompt` or `Terminal`, and type the following which will display your python version. 

```
python --version
```
<br>

### Create a virtual environment

Open `Command Prompt` or `Terminal`. Then, navigate to folder where you want to create the virtual environment. For example, 

```
# windows users
cd C:\Users\midani\apps\

# mac/linux users
cd /Users/midani/apps
```

Create environemnt

```
python -m venv amiga-env
```

Activate the environment

```
# windows users
amiga-env\Scripts\activate

# mac/inux suers
source amiga-env/bin/activate
```
<br>

### Install AMiGA
Type the following in your `Command Prompt` or `Terminal`

```
pip install amiga
```

Verify that installation worked

```
amiga -h
```

If everything is correc,t you will see the main `amiga` help menu. 

```
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
<br>

### Running AMiGA in the future
Open `Command Prompt` or `Terminal`. Then, navigate to where you installed the virtual environment

```
cd C:\Users\midani\apps\
```

Activate the environment

```
# windows users
amiga-env\Scripts\activate

# mac/inux suers
source amiga-env/bin/activate
```

Now, you can run AMiGA

```
amiga --help
```

\[Optional\] Deactivate when done working with AMiGA.

```
deactivate
```

