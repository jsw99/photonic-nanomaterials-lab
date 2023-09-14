# Cell segmentation
## Table of contents
* [General info](#general-info)
* [Workflow](#workflow)
* [Results](#results)

## General info
We aim at counting the fluorescent intensity of SWCNTs incubated in each cell, which will lay the foundation of further analysis.
For cellular segmentation, we use the API from [Pachitariu, M. & Stringer, C. (2022). Cellpose 2.0: how to train your own model. Nature methods, 1-8.](https://github.com/mouseland/cellpose)

## Workflow
The general worflow is described below:
### 1. Read the input brightfield and fluorescence microscope images
### 2. Perform cell segmentation on the brightfield images
### 3. Convert outlines into cell boundaries
### 4. Calculate fluorescence intensity in each segmented cell

## Results
