---
layout: page
title: "Organization"
category: doc
date: 2020-03-24 19:33:36
---

Before you begin with your first analysis, you need to create a folder which we will refer to as the __Working Directory__. Here, I name my example as __experiment__. In this folder, you must have a __data__ sub-folder where you will save all of your data files that need to be analyzed.

<br /><br />

![amiga directory tree example](../assets/img/amiga_directory_tree_simple.png){:width="500px"}
<!--- ![amiga directory tree example](../assets/img/amiga_directory_tree_simple.png){:class="img-responsive"} --->

<br /><br />

`AMiGA` is designed for the analysis of plate reader data. Plate readers typically measure the optical density in every well of a 96-well microplate at fixed time intervals (e.g. every 10 minutes). These files are often saved as `.TXT` or `.ASC` files and may contain run information such as protocol, temperature, ... etc. in the first few lines of the text file. After the run information, data will typically look like the following example.

<br /><br />

![example data file](../assets/img/example_data_file.png){:width="300px"}
<!--- ![amiga directory tree example](../assets/img/amiga_directory_tree_simple.png){:class="img-responsive"} --->

<br /><br />

The first column is the well location which is a unique identifier of each well. The  first row is the time measurement. The cell values are the measured optical density in each well at a specific time point.


__Frequently Asked Questions (FAQ)__

*Must be the input be a 96-well plate? Can it be a 384-plate? How about just a couple of wells or even one?*

No, the input does not have to conform to 96-well format. It can by any number of wells.

*Must the index column be well IDs? Can it be something else?*

Yes. 
