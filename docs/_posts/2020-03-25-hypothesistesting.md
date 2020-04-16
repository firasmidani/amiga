---
layout: page
title: "Hypothesis Testing"
category: doc
date: 2020-03-25 22:44:58
order: 8
use_math: true
---

`AMiGA` can perform Gaussian Process (GP) regression to test differential growth between distinct experimental conditions.

<br/>

**Example One**

```bash
python main.py -i /home/outbreaks/erandomii/ -s 'Isolate:ER1;PM:1;Substrate:Negative Control,alpha-D-glucose' -y 'H0:Time;H1:Time+Substrate'
```

Here, we first reduced our data set only to the growth curves of the isolate `ER1` in `PM1` plates on `Negative Control` (i.e. no carbon) and `alpha-D-glucose` wells. We then test the null hypothesis (`H0`) that only `Time` variable explains variation in OD against the alternative hypothesis (`H1`) that both `Time` and `Substrate` variables explain the variation in OD.

`AMiGA` will process your request and create four files in the `models` folder.

- `key`: reduced mapping file for your request.
- `input`: data used for hypothesis testing (e.g. Time, OD, ..., etc)
- `output`: summary of the the results of the GP regression test.
- `pdf`: plot of the data tested with GP Regression.

**Limitations of hypothesis Testing:**

- Testing can only be performed to compare two conditions. In the above example, we compared the grown on the glucose well against growth on the control well.
- If you are comparing data spread across multiple plates, there will be batch effects. To account for this, `AMiGA` will subtract the grwoth in control wells from each of your growth cures. To over-ride this ,you can the `--dont-subtract-control` argument.
 
