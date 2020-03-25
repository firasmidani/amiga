---
layout: page
title: "Organization"
category: doc
date: 2020-03-24 19:33:36
---

**Table of Contents**

* TOC
{:toc}
<br />

### Organizing your working directory

Before you begin with your first analysis, you need to create a folder which we will refer to as the __Working Directory__. Here, I name my example as __experiment__. In this folder, you must have a __data__ sub-folder where you will save all of your data files that need to be analyzed.

<br />

![amiga directory tree example](../assets/img/amiga_directory_tree_simple.png){:width="500px"}
<!--- ![amiga directory tree example](../assets/img/amiga_directory_tree_simple.png){:class="img-responsive"} --->

<br />

### Preparing your plate reader data files

`AMiGA` is designed for the analysis of plate reader data. Plate readers typically measure the optical density in every well of a 96-well microplate at fixed time intervals (e.g. every 10 minutes). These files are often saved as `.TXT` or `.ASC` files and may contain run information such as protocol, temperature, ... etc., in the first few lines of the text file. After the run information, data will typically look like the following example. Here, the first column is the well location which is a unique identifier of each well. The  first row is the time measurement. The cell values are the measured optical density in each well at a specific time point.

<br />

![example data file](../assets/img/example_data_file.png){:width="300px"}
<!--- ![amiga directory tree example](../assets/img/amiga_directory_tree_simple.png){:class="img-responsive"} --->

<br />

Plate readers use different proprietary softwares that export data in slightly different formats. To avoid confusion due to different formats, `AMiGA` will ignore the time row and will not read it. Instead, it will detect the first line that starts with a well location (e.g. A1 or D13) and read all subsequent lines. Other lines in the text file will simply be ignored. `AMiGA` will instead rely on the `Interval` parameter to define the time points. By default this value is 600 seconds (or 10 minutes) but the user can over-ride default in the `config.py` file or by passing another `Interval` value as an argument. See PLACEHOLDER section for details.

<br /><br />

### Naming your plate reader data files

Please use only alphanumeric characters in your file names.

If you are analyzing Biolog Phenotype Microarray (PM) plates, you can name your data file in a specific way and ```AMiGA``` will automatically recognize it as a Biolog PM plate, identify the content in each well on your plate, and include these details in the summary of your growth curves. To do so, please use this nomenclature:

\{isolate name\}_PM\{integer\}-\{integer\}

where the isolate name can be composed of any alphanumeric characters, the first integer indicates the PM number (must be between and including 1 and 7), and the second integer indicates the replicate number. For example, `CD630_PM3-2.txt` corresponds to the second replicate of growing the isolate `CD630` on `PM3`.

<br /><br />

### Frequently Asked Questions (FAQ)

__Which file formats AMiGA read?__

AMiGA can only read tab-delimited text files which is the typical format for output by plate readers. These can be encoded in ASCII or BOM (e.g. UTF-8).

__Can AMiGA read a Microsoft Excel files?__

No, please convert the file to tab-delimited text file format.

__Must the input be a 96-well plate? Can it be a 384-well plate? How about just a couple of wells/rows or even a single well/row?__

No, the input does not have to conform to a 96-well format. It can describe results for any number of wells/rows.

__Must the index column be well IDs? Can it be something else?__

The index column (or row names) must be well locations (e.g. D13) where first character is an alphabetic letter that corresponds to a specific row in the plate and remaining characters are digits that corresponds to a specific column in the plate
