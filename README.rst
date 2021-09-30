
.. image:: material/logo.png
    :width: 350


.. image:: https://img.shields.io/pypi/v/pythonpredictions-cobra.svg
    :target: https://pypi.org/project/pythonpredictions-cobra/
.. image:: https://img.shields.io/pypi/dm/pythonpredictions-cobra.svg
    :target: https://pypistats.org/packages/pythonpredictions-cobra
.. image:: https://github.com/PythonPredictions/cobra/actions/workflows/development_CI.yaml/badge.svg?branch=develop
    :target: https://github.com/PythonPredictions/cobra/actions/workflows/development_CI.yaml

------------------------------------------------------------------------------------------------------------------------------------ 

**Cobra** is a Python package to build predictive models using linear or logistic regression with a focus on performance and interpretation. It consists of several modules for data preprocessing, feature selection and model evaluation. The underlying methodology was developed at Python Predictions in the course of hundreds of business-related prediction challenges. It has been tweaked, tested and optimized over the years based on feedback from clients, our team, and academic researchers.

Main features
=============

- Prepare a given pandas DataFrame for predictive modelling:

   - partition into train/selection/validation sets
   - create bins from continuous variables
   - regroup categorical variables based on statistical significance
   - replace missing values
   - add columns where categories/bins are replaced with average of target values (linear regression) or with incidence rate (logistic regression)
 
- Perform univariate feature selection based on RMSE (linear regression) or AUC (logistic regression)
- Compute correlation matrix of predictors
- Find the suitable variables using forward feature selection
- Evaluate model performance and visualize the results

Getting started
===============

These instructions will get you a copy of the project up and running on your local machine for usage, development and testing purposes.

Requirements
------------

This package requires only the usual Python libraries for data science, being numpy, pandas, scipy, scikit-learn, matplotlib, seaborn, and tqdm. These packages, along with their versions are listed in ``requirements.txt`` and can be installed using ``pip``:    ::

  pip install -r requirements.txt


**Note**: if you want to install Cobra with e.g. pip, you don't have to install all of these requirements as these are automatically installed with Cobra itself.

Installation
------------

The easiest way to install Cobra is using ``pip``:    ::

  pip install -U pythonpredictions-cobra


Documentation and extra material
=====================

- A `blog post <https://www.pythonpredictions.com/news/the-little-trick-we-apply-to-obtain-explainability-by-design/>`_ on the overall methodology.

- A `research article <https://doi.org/10.1016/j.dss.2016.11.007>`_ by Geert Verstraeten (co-founder Python Predictions) discussing the preprocessing approach we use in Cobra.

- HTML documentation of the `individual modules <https://pythonpredictions.github.io/cobra.io/docstring/modules.html>`_.

- A step-by-step `tutorial <https://pythonpredictions.github.io/cobra/tutorials/tutorial_Cobra_logistic_regression.ipynb>`_ for **logistic regression**.

- A step-by-step `tutorial <https://pythonpredictions.github.io/cobra/tutorials/tutorial_Cobra_linear_regression.ipynb>`_ for **linear regression**.

- Check out the Data Science Leuven Meetup `talk <https://www.youtube.com/watch?v=w7ceZZqMEaA&feature=youtu.be>`_ by one of the core developers (second presentation). His `slides <https://github.com/PythonPredictions/Cobra-DS-meetup-Leuven/blob/main/DS_Leuven_meetup_20210209_cobra.pdf>`_ and `related material <https://github.com/PythonPredictions/Cobra-DS-meetup-Leuven>`_ are also available.

Contributing to Cobra
=====================

We'd love you to contribute to the development of Cobra! There are many ways in which you can contribute, the most common of which is to contribute to the source code or documentation of the project. However, there are many other ways you can contribute (report issues, improve code coverage by adding unit tests, ...).
We use GitHub issue to track all bugs and feature requests. Feel free to open an issue in case you found a bug or in case you wish to see a new feature added.

For more details, check our `wiki <https://github.com/PythonPredictions/cobra/wiki/Contributing-guidelines-&-workflows>`_.
