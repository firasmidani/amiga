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
- If you are comparing data spread across multiple plates, there will be batch effects. To account for this, `AMiGA` will subtract the grwoth in control wells from each of your growth curves. To over-ride this, you can use the `--dont-subtract-control` argument.

**Interpretation of model output**

See [Example](/amiga/doc/example.html) for more details on the context of this example.

The figure will show the models estimated for each strain (bold lines) overlaid on the actual data (thin lines). Shaded bands indicate the 95% confidence interval for the models.

![lactic acid figure](../assets/img/strain_difference_l_lactic_acid.png){:width="400px"}

<br />

The output report will look like this:

```
The following criteria were used to subset data:
Substrate......['L-Lactic Acid']

The following hypothesis was tested on the data:
{'H0': ['Time'], 'H1': ['Time', 'Strain']}

log Bayes Factor: 750.772 (0.0-percentile in null distribution based on 100 permutations)

For P(H1|D) > P(H0|D) and FDR <= 20%, log BF must be > 0.179
For P(H0|D) > P(H1|D) and FDR <= 20%, log BF must be < -4.34e-06

Data Manipulation: Input was reduced to 34 time points. Samples were normalized to relevant control samples before modelling.
```

This indicates that the log Bayes Factor is 750.772 and much higher than the 20% FDR threshold of 0.179. You are every confident that Lactic Acid supports the growth of the MEX_2020 strain but not the USA_1995 strain.

Recall, that

$$\text{Bayes Factor} = \exp{\log \text{Bayes Factor}} = \exp{750.722} = 1.44$$

and

$$\text{Bayes Factor} = \frac{P(H1|D)}{P(H0|D)}$$

Therefore, the analysis suggest that alternative hypothesis that strain differences contributes to differences in growth is more supported than the null hypothesis that only time explains variations in optical density measurements. However, because the FDR is set to 20% and because the estimated models do not visually differ, you believe these minute differences detected by the GP regression are more likely explained by batch effects (i.e. random noise).
