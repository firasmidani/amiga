---
layout: page
title: "Hypothesis-Testing"
category: doc
date: 2020-03-25 22:44:58
order: 8
use_math: true
---

`AMiGA` can perform Gaussian Process (GP) regression to test differential growth between distinct experimental conditions.

<br/>

**Example One**

```bash
python main.py -i /home/outbreaks/erandomii/ER_PM1-1.txt -s 'Substrate:Negative Control,alpha-D-glucose' -h 'H0:Time;H1:Time+Substrate'
```
