# Global_vegetation_water_sensitivity
This repository contains two parts for a paper to be published as: Li, W., Migliavacca, M., Forkel, M., Denissen, J.M.C., Reichstein, M., Yang, H., Duveiller, G., Weber, U. and Orth, R. (2022). Widespread increasing vegetation sensitivity to soil moisture (Under review).

i) Demo codes and data required to calculate overall and 3-year-block LAI sensitivities to soil moisture;

ii) Codes required to reproduce main figures in the paper. The raw datasets mentioned below are used to compute the analysis results. These analysis results are required to produce the final figures and are stored at 'zenodo link' (as the data size is big).

We use the demo for transparency and to demonstrate the application of sensitivity analysis using machine learning.

#### We are happy to answer your questions! Contact: Wantong Li (wantong@bgc-jena.mpg.de) 

### Conda environment installation
Please use the base.yml to set up the environment for runing provided codes. The Linux command for environment installation: conda env create -f base.yml

### The guide of demo_of_sensitivity_analysis
i) To save the runtime, one satellite LAI product and one soil moisture reanalysis are used as an example, while for the paper analysis we calculate many times of sensitivity results using different satellite and land surface modelled LAI and soil moisture products;

ii) To save the runtime, European domains are used instead of the global scale when calculating sensitivity results.

### The guide of figure_codes
i) All processed data are in NumPy array format for Python;

ii) Original data are shared with public links in the paper. 

### References
i) Random forest modelling refers to: 

Breiman, L. Random forests. Machine Learning 45, 5–32 (2001).

ii) SHAP values to interpret contributions of predictors to target variables refers to:

Lundberg, S. M. & Lee, S. I. A unified approach to interpreting model predictions. (2017);

Molnar, C. Interpretable Machine Learning: A Guide for Making Black Box Models Explainable (2021). [online: https://christophm.github.io/interpretable-ml-book/]
