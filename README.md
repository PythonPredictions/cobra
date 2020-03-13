# COBRA :snake: <img src="https://github.com/JanBenisek/Pytho/blob/master/pythongrey%20large.png" width="100" align="right">

**Cobra** is a Python package that implements the Python Predictions methodology for predictive analytics. It consists of a main script/notebook that can be used to build and save a predictive model only by setting several parameters. The main scripts itself consists of several modules that can be used independently of one another to build custom scripts.

Note that this package is a refactored version of the back-end of the original web-based cobra.

:heavy_exclamation_mark: Be aware that there could still be :bug: in the code :heavy_exclamation_mark:

## What can cobra do?

  * Prepare a given pandas DataFrame for prediction modelling:
    - partition into train/selection/validation sets
    - create bins from continuous variables
    - regroup categorical variables
    - replace missing values and
    - add columns with incidence rate per category/bin.
  * Perform univariate selection based on AUC
  * Compute correlation matrix of predictors
  * Find best model by forward selection
  * Visualize the results
  * Allow iteration among each step for the analyst

## Getting started

These instructions will get you a copy of the project up and running on your local machine for usage, development and testing purposes. Furthermore, this section includes some brief examples on how to use it.

### Requirements

This package requires the usual Python packages for data science:

* numpy
* scipy
* matplotlib
* seaborn
* pandas
* scikit-learn

These packages, along with their versions are listed in `requirements.txt` and `conda_env.txt`. To install these packages using pip, run

```
pip install requirements.txt
```

or using conda

```
conda install requirements.txt
```

### Installation

As this package is an internal package that is not open-sourced, it is not available through `pip` or `conda`. As a result, the package has to be installed manually using the following steps:

  * Clone this repository.
  * Open a shell that can execute python code and navigate to the folder where this repo was cloned in.
  * Once you are in the folder, execute `python setup.py install` or `pip install .`.

### Usage

TO DO

## Development

We'd love you to contribute to the development of Cobra! To do so, clone the repo and create a _feature branch_ to do your development. Once your are finished, you can create a _pull request_ to merge it back into the main branch. Make sure to write or modify unit test for your changes!
