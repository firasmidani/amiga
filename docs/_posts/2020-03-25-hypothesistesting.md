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
python main.py -i /home/outbreaks/erandomii/ -s 'Isolate:ER1;PM:1;Substrate:Negative Control,alpha-D-glucose' -h 'H0:Time;H1:Time+Substrate'
```

Here, we first reduced our data set only to the growth curves of the isolate `ER1` in `PM1` plates on negative control (i.e. no carbon) or alpha-D-glucose wells. We then test the null hypothesis that only time variable explains variation in OD against the alternative hypothesis that both time and substrate explain the variation in OD.
