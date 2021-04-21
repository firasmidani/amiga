---
layout: page
title: "Configuration"
category: doc
date: 2021-04-18 15:54:41
use_math: true
order: 18
---

**Table of Contents**

* TOC
{:toc}
<br />

<!-- AMiGA is covered under the GPL-3 license -->
 
`AMiGA` has many parameters that can be configured by the user. Some of those parameters are configured using the [Command-line interface](/amiga/doc/command-line-interface.html). Other parameters can be configured in the `libs/config.py` file. These are parameters that define user preferenecs for common operations and it would be impractical to ask the user to define them every time they submit a command to `AMiGA`. The `libs/config.py` defines the default values for these parameters. Here, I elaborate on these parameters and explain how users can adjust them. 

The `libs/config.py` is a `Python` script that define a single `dictionary` called `config`. You can open this `Python` script in a text editor. If you are not familiar with dictionaries, it a basic data structure that operates similar to a word dictionary. In a `dictionary`, there are a set of `keys` and for each key there is a `value` (i.e. definition). The values of the `config` dictionary are esentially defining the set of parameters (or keys) used by `AMiGA`. 

The parameters that the user can configure can be divided into several groups based on their purpose:
- Data input and manipulation
- 96-well grid plotting
- GP regression modelling
- Hypothesis testing plotting
- Growth parameter definitions
- User communication

<br />
#### Data input and manipulation: Time measurements


`time_input_unit` and `time_output_unit` and `interval`: While AMiGA does not read the time points in the input plate reader files, it does require the user to define the time interval between measurements. . Users can define the time interval via the command-line with the `--interval` argument. But if the user often uses the same time interval for their growth assays, they can simply define the default `interval` in the `libs/config.py` file as well as the `time_input_units`. For example, if OD measurements are taken every 30 minutes. The user can do either

```python
config['interval'] = 1800
config['time_input_unit'] = 'seconds'
```

or 

```python
config['interval'] = 30
config['time_input_unit'] = 'minutes'
```

or 

```python
config['interval'] = 0.5
config['time_input_unit'] = 'hours'
```

The `time_output_unit` basically tells `AMiGA` how to convert the time interval which affects how the results (growth parameters, growth curves, and plots) are interpreted. 

<br />
#### Data input and manipulation: How to center the growth curve on zero?

After several pre-processing steps, `AMiGA` transforms growth curves with a natural logarithm. The, `AMiGA` needs to adjust the growth curve so that the first OD measurement is near zero. The default approach is to simply subtract the first OD measurement from all other measurements in the growth curve (i.e., $$\text{Adjusted OD}(t) = \text{OD}(t) - \text{OD}(0)$$). When users are fitting replicate growth curves together, an alterantive approach is called `PolyFit`. Here, `AMiGA` will fit first five OD measurements with a third-degree polynomial. Based on the polynomial fit, it will estiamte the OD measurement at the first time point then subtract this estimated OD measurement from the other OD measurements in the growth curve (i.e., $$\text{Adjusted OD}(t) = \text{OD}(t) - f(t=0)$$ where $$f(t)$$ is the polynomial fit. By default, `AMiGA` uses the similar approach of subtracting the first OD measurement but users can opt to use `PolyFit` instead.

```python
config['PolyFit'] = True
```

<br />
#### Data input and manipulation: Handling non-positive measurements?

Optical density measurement are supposed to be non-zero and positive values. However, there are cases where plate readers can measure negative or zero values. This is especially common if the OD measuremnets are blank-corrected. In these cases, the OD measurements over time for blank media is subtracted from the OD measurements of treatment wells where you expect to see growth. `AMiGA` can handle non-positive values using two ways: the limit of detection `LOD` method or the optimal offset `Delta` method. 

The `LOD` method is premised on the fact that all plate readers have a limit of detection. This may be either the lowest optical density measurement it can accurately detect or the smallest increments of optical density that it can accurately detect. If you know this value, you can basically assume that an optical density of value zero should be raised to the limit of detection. But growth curves can also be negative (as before-mentioned due to things like blank subtraction). So, `AMiGA` first translates vertically the whole growth curve such that the lowest non-positive value becomes zero then raises all the measurements by the limit of detection. In other words, $$\text{Adjusted OD}(t) = \text{OD}(t) + \mid\min \text{OD}(t)\mid + \text{LOD}$$. Users can also force the LOD correction for all measurement such that no OD measurement is lower than the limit of detection. 

To use the `LOD` method, you can adjust the following parameters. The first selects `LOD` as your method of choice; the second defines the limit of detection; and the third determines if you would like the floor for all your measurements to become the limit of detection. 

```python
config['handling_nonpositives'] = 'LOD'
config['limit of detection'] = 0.010
config['force_limit_of_detection'] = False
```


The `Delta` method assumes that you do not know the limit of detection and would rather use the data to infer a reasonable offset. In a similar fashion, here, `AMiGA` wants to vertically translate the whole growth curve such that the lowest non-positive value becomes zero then raise all the measurements buy the estimated offset. The offset is determined based on the distribution of actual changes in OD over time. In particular, `AMiGA` computes the difference in OD between all or some consecutive time intervals. It can then pick the OD based one statistical descriptor of this distribution. 

<br />

To use the `Delta` method, you can adjust the following parameter. The first selects `Delta` as your method of choice; the second determienes the number of time intervals to use for determing the optimal delta; and the third determines which delta to pick from the distribution

```python
config['handling_nonpositives'] = 'Delta'
config['number_of_deltas'] = 5
config['choice_of_deltas'] = 'median'
```

For example, the above parameters state the change in OD for the first five time intervals will be computed and the optimal delta is considered their median. If the user would like to use the whole growth curve for computing delta, they can simply pick a very large number like 1000. This way they don't have to adjut the number for growth curves with different number of time measurements. 

On a final note, if you choose the `LOD` method but a particular growth curve has a negative measurement that has an absolute vlaue larger than the limit of detection, then `AMiGA` will be forced to use the `Delta` method. 

<br />
#### 96-well grid plotting

The `summarize` and `fit` function of `AMiGA` can plot the growth curves for 96-well plate reader files in an crisp PDF format. The user can adjust several parameters for the aesthetics of these plots. 

`fcg` or fold-change threshold for growth and `fcd` or fold-change threshold for death dictate how each growth cuve will be colored. If the user includes in the meta-data which wells correspond to control and which correspond to cases or treatments, `AMiGA` will automatically compute the `Fold-Change` (See [Summarize and Plot](/amiga/doc/plotting.html) for more details). Here, the parameters define the threshold at which thes growth curve would be colored: if a well has a fold-change greater than 1.5 or less than 0.5 the its curve will be colored. 

```python
config['fcg'] = 1.50  # fold-change threshold for growth
config['fcd'] = 0.50  # fold-change threshold for death
```

<br />
The colors can be defined with the following parameters. Here, I define the colors in (R,G,B,A) format where A is a value that adjusts the transparency fo the color. All values range from 0.0 to 1.0 (and map to 0 and 255 in decimal notation). However, you can can also define colors with text label (e.g., 'red') or hex stirn format (e.g., '#0099CC'). See [List of named colors](https://matplotlib.org/stable/gallery/color/named_colors.html) for a long list of colors that you can use with `Python`. 

```python
config['fcg_line_color'] = (0.0,0.0,1.0,1.0)  # blue
config['fcg_face_color'] = (0.0,0.0,1.0,0.15) # transparent blue

config['fcd_line_color'] = (1.0,0.0,0.0,1.0)  # red
config['fcd_face_color'] = (1.0,0.0,0.0,0.15) # transparent red

config['fcn_line_color'] = (0.0,0.0,0.0,1.0)  # black
config['fcn_face_color'] = (0.0,0.0,0.0,0.15) # transparetn black

config['gp_line_fit'] = 'yellow'
```

`fcg_line_color` and `fcg_face_color` define the color for the line and the area of the growth curve where growth is detected based on fold-change; `fcd_line_color` and `fcd_face_color` likewisedefine colors for wells where death is detected; and `fcn_line_color` and `fcn_face_color` define colors for the remaining wells. The last parameter `gp_line_fit` defines the color for the dashed line that plots the curves predicted by Gaussian Process regression. These lines can only be plotted by the `fit` function. 

<br />
In these plots, `AMiGA` also adds text in each well for the well ID (e.g. "A1" on top left corner) or Maximum OD (on top right corner). Users can also adjus the color for these wells in a similar fashion

```python
config['fcn_well_id_color'] = (0.65,0.165,0.16,0.8) # light maroon
config['fcn_od_max_color'] = (0.0,0.0,0.0,1.0)      # black

```

<br />

Finally, users can adjust the label for the y-axis. For the `fit` function, the generated plot will be based on growth curves that are log-transformed. So the final label would look something like this "ln Optical Density". `AMiGA` was designed to analyze optical density data but it can model any count data (e.g. "CFUs" or "fluorescence"). By adjusting the y-axis label, users can distinguish the type of data that they are looking at. 

```python
config['grid_plot_y_label'] = 'Optical Density'
```

<br />
#### Hypothest test ploting

The `test` function in `AMiGA` can generate figures that plot the predicted fit for the two growth curves being compared. Users can adjust the colors for the lines, y-axis label, and several parameters about aesthetics of the figure. 

```python
config['hypo_colors']  = [(0.11,0.62,0.47),(0.85,0.37,0.01)]  # seagreen and orange

config['hypo_plot_y_label'] = 'OD'

config['HypoPlotParams'] = {'overlay_actual_data':True,
			    'fontsize':15,
			    'tick_spacing':5,
			    'legend':'outside'}
```

The default colors for the lines correspond to seagreen and orange and the default label is "OD". Users can also adjust the fontsize (in points), the spacing of the x-axis ticks. Here, the `tick_spacing` is defined as `5` and because our default is `time_output_interval` is `hours`, `AMiGA` will plot ticklabels at intervals of `5 hours`. By default, the `legend` is plotted `outside` but can instead be plotted `inside`. Finally, users can opt to `overlay_actual_data` or turn off this feature with `False`. If `True`, `AMiGA` will plot the predicted growth curve in bold line and the raw growth curves in thin lines. All growth curves will however be log-transformed and centered at zero (because ln OD = 0 &#8594; OD=1, starting value of zero in log OD indicates arbitrary starting population size of one).

<br />
#### Growth parameter estimation and reporting

One of advantages of inferring the growth rate over time is that `AMiGA` can identify the time at which growth rate begins. It does so by identifying the time point at which the growth rate is statistically different from a growth rate of zero. We refer to this `Adaptation Time` as the time at which the confidence itnerval of the growth rate deviates from zero. Users can adjust the confidence for this interval.

```python
config['confidence_adapt_time'] = 0.95
```

<br />
Another unique feature for `AMiGA` is its novel algorithm for detecting and characterizing diauxic shifts. See [Detect Diauxie](/amiga/doc/diauxic-shift-detection.html) for more details on the parameters.

```python
config['diauxie_ratio_varb'] = 'K' 
config['diauxie_ratio_min'] = 0.20 
config['diauxie_k_min'] = 0.10
```

<br />

By default `AMiGA` simply provides the mean estimate for each growth parameter (e.g. growth rate). However, if the user passes the `--sample-posterior` argument to the `fit` function, `AMiGA` will also generate distributions for these parameters, in particular the mean and standard deviation. To do so, `AMiGA` samples from the posterior distribution of the growth model using a certain number of sampled curves. It computes the growth rates for each of those curves and then reports the mean and standard deviation of these distributions. Users can adjust how many posterior samples are drawn but keep in mind that the more samples drawn the longer the process will take. 

```python
config['n_posterior_samples'] = 100
```

<br />

Finally, `AMiGA` can infer up to 15 growth parameters but if the user would like the summary files to include only the ones they are interested in, they can simply alter the list below to their liking. See [Fit Curves](/amiga/doc/fitting.html) for descriptions of these parameters.

```python
config['report_parameters'] = ['auc_lin','auc_log','k_lin','k_log','death_lin','death_log',
			       'gr','dr','td','lagC','lagP',
                               't_k','t_gr','t_dr','diauxie']
```

<br />
#### Estimating variance or goodness of fit

If the user opts to empirically-estimate time-dependent Gaussian noise, `AMiGA` will compute the variance of replicate data over time. It will then apply a Gaussian filter to smoothen the variance over time. Users can adjust the window for smoothign variance. The defaul is `6` measurements, which correspond to 1 hour (defaul interval is 600 seconds, so 6 x 6 = 3600 seconds = 1 hour). 

```python
config['variance_smoothing_window'] = 6 
```

<br />

As a quick check of the goodness of fit of predicted growth curves, `AMiGA` computes the `K_Error` which estiamtes the deviation of the predicted carrying capacity from the expected carrying capacity. See [Fit Curves](/amiga/doc/fitting.html) for descriptions of these parameters. Users can adjust the threshold at which `AMiGA` will flag a well if it has a `K_Error` above this threshold. The default is `20%`.  

```python
config['k_error_threshold'] = 20
```

<br />
#### User communication

`AMiGA` implements Gaussian Process regression using the `GPy` package. Many functions and features of this package often communicate warnings. By default, `AMiGA` will not show these warnings to the user but if you would like to see them, you can simply turn off the below parameter wtih `False`. 

```python
config['Ignore_RuntimeWarning'] = True
```

<br />