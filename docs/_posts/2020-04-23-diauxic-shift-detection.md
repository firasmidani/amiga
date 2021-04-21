---
layout: page
title: "Detect Diauxie"
category: doc
date: 2020-04-23 12:10:16
use_math: true
order: 8
---
<!-- AMiGA is covered under the GPL-3 license -->
**Table of Contents**

* TOC
{:toc}

<br />

#### What is diauxie?

Diauxic growth or diauxie is characterized by multiple exponential growth phases in a bacterial culture. These growth phases are characterized by different growth dynamics and often separated by lag phases. These shifts tend to occur due to a change in the composition of the growth media, for example, when a microbial culture exhausts a preferred nutrient or compound in the environment then switches to a less preferred nutrient or compound.

<br />
#### How does `AMiGA` detect diauxie?

`AMiGA` can detect those diauxic shifts (or any multi-phasic growth) when fitting growth curves. See [Midani et al. (2020)](https://www.biorxiv.org/content/10.1101/2020.11.04.369140v1) for description of the algorithm. The algorithm can be fine-tuned using three different parameters that can be adjusted in `libs/config.py` file (see [Configure default parameters](/amiga/doc/configuration.html)).

- `diauxie_ratio_varb`: dictates which variable to use for detection of multiple phases of growth. Select `K` for using carrying capacity or `r` for using growth rate. The default  vaiable is `K`.

- `diauxie_ratio_min`: AMiGA determines whether a potential growth phase is an actual growth phase by comparing either their carrying capacities or growth rates. Here, this paramter defines the minimum ratio of the growth or growth rate of the secondary peak, relative to the primary peak; this ratio determines whether a potential growth phase is called as an actual growth phase. The default is that secondary peaks must have at least 10% or higher `K` or `r` relative to the primary peak. 

- `diauxie_k_min`: Only analyze growth curves where total growth, `ln(OD)`, is above this threshold. The default is 0.10. 

<br />
#### Where does `AMiGA` report diauxic shifts and related parameters?

`AMiGA` users do not need to explicitly request diauxic shift detection. `AMiGA` automatically detects these shifts when fitting growth curves. If multi-phasic growth was detected in a well, the `summary` text file will indicate it with a value of `1` in the `diauxie` column. Further, the `diauxie` text file will include all growth parameters for each unique growth phase detected for these wells. Growth phases are reported in chronological order. Gowth parametes are given the prefix `dx_` and include `dx_t0` and `dx_tf` which respectively indicate the initial and final time points for the corresponding growth phase. 
