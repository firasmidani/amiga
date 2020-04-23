---
layout: page
title: "Diauxic Shift Detection"
category: doc
date: 2020-04-23 12:10:16
use_math: true
order: 9
---

Diauxic growth or diauxie is characterized by multiple exponential growth phases in a bacterial culture. These growth phases  are characterized by different growth rates and separated by secondary lag phases. These shifts tend to occur due to a change in the composition of the growth media, for example when a microbial culture exhausts a preferred nutrient or compound in the environment then switches to a less preferred nutrient or compound.

`AMiGA` can detect those diauxic shifts when fitting growth curves. It first computes the derivative of the Optical Density (OD) over time, i.e. change in the OD over change of time. Each peak in the derivative indicates a local maximum growth rate. `AMiGA` will determine those peaks and only classify them as diauxic shifts if they meet certain criteria.

First, all secondary growth phases must have results in a change in OD that is at least a certain ratio relative to the change in OD due to the primary growth phase. The default value is defined `0.20` (i.e. 20% of the height of the primary growth phase) in the `libs/config.py` file.

Second, `AMiGA` is sensitive enough to detect any peak in the OD data. However, it will classify those peaks as diauxic shifts if they meet the above criteria and if the growth data experience a certain fold-change relative to the sample control. This fold-change minimum is defined with a default value of `1.5` which can be changed in the `libs/config.py` file.

> Note: If your plate lacks any control samples or your mapping files do not indicate which samples are control, diauxie shift detection will result in an error. This issue will be resolved in a new update.

`AMiGA` users do not need to explicitly request diauxic shift detection. `AMiGA` automatically detects these shifts when fitting growth curves. The results are reported in the summary text files using 2 or more columns.

- `diauxie` column is a binary variable that indicates if secondary growth phases were dected (1 means Yes; 0 means No).

- `t_1_peak` column indicates the time to reach mid-exponential growth for the primary phase.

- `t_2_peak` column indicates the time to reach mid-exponential growth for the secondary phase.

- additional third, fourth, and so on, will also be described in additional columns labeled `t_n_peak` where `n` indicates the growth phase.

Growth phases are reported in the order of their relative heights. Therefore, the primary growth phase that results in the largest change in OD and may not be necessarily the first growth phase.


PLACEHOLDER FOR EXAMPLE PLOT AND RESULTS
