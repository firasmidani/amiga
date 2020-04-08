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

Here, we first reduced our data set only to the growth curves of the isolate `ER1` in `PM1` plates on `Negative Control` (i.e. no carbon) or `alpha-D-glucose` wells. We then test the null hypothesis (`H0`) that only `Time` variable explains variation in OD against the alternative hypothesis (`H1`) that both `Time` and `Substrate` variables explain the variation in OD.
