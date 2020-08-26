---
layout: page
title: "Batch Analysis"
category: doc
date: 2020-03-25 22:44:43
order: 6
use_math: true
---
<!-- AMiGA is covered under the GPL-3 license -->

In `AMiGA`, you can analyze a single file or multiple files in a single command. To analyze a single file, simply point to it directly with the `--i` or `--input` arguments

```bash
python amiga.py -i /Users/firasmidani/experiment/data/ER1_PM2-1.txt
```

Bug if you want to analyze multiple files, simply deposit them in the `data` folder and point `AMiGA` to the working directory.

```bash
python amiga.py -i /Users/firasmidani/experiment
```
