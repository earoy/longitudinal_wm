# longitudinal_wm

This repository contains the Python code to load the data, run the modeling pipelines, and generate the figures preesented in the manuscript. Tract profiles for the HBN dataset can be found [INSERT LINK HERE]. Information about accessing the phenotypic data for HBN can be found [here](http://fcon_1000.projects.nitrc.org/indi/cmi_healthy_brain_network/Pheno_Access.html). To access the PING and ABCD data, you need to request access to the [NIMH Data Archieve](https://nda.nih.gov/get/access-data.html) and the [ABCD study](https://nda.nih.gov/abcd). Once you have access to those data, you can clone repository, create a **/data** directory, and run the analysis scripts. 

## Repository Organization

* **/notebooks** contains two notebooks: **load_and_harmonize_data.ipynb** and **figures.ipynb**. The first notebook loads the neuroimaging and phenotypic data across the three datasets and merges them into dataframes that can later be used to make group-wise comparisons or in XGBoost modeling. The second notebook loads the merged datasets and outputs from XGBoost models to generate the figures found in the manuscript.  
* **/xgb_scripts** contain three Python scripts for running XGBoost models on the HBN, ABCD, and PING datasets. These scripts take the outputs from the **load_and_harmonize_data.ipynb** notebook as inputs and export .csv files with model scores and predictions for use with the **figures.ipynb** notebook.
* **/data** contains the data for running various analyses and generating figures. This directory should have a sub-directory for each study (HBN, ABCD, PING) that contains the neuroimaging and phenotypic data files. 
* **/utils** contains utility functions for extracting features from the various dataframes, as well as merging neuroimaging and phenotypic dataframes. 




