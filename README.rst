

.. image:: https://img.shields.io/pypi/v/pythonpredictions-cobra.svg
    :target: https://pypi.org/project/pythonpredictions-cobra/
.. image:: https://img.shields.io/pypi/dm/pythonpredictions-cobra.svg
    :target: https://pypistats.org/packages/pythonpredictions-cobra
.. image:: https://github.com/PythonPredictions/cobra/actions/workflows/development_CI.yaml/badge.svg?branch=develop
    :target: https://github.com/PythonPredictions/cobra/actions/workflows/development_CI.yaml

------------------------------------------------------------------------------------------------------------------------------------ 

=====
cobra
=====

.. image:: material\logo.png
    :width: 300

**cobra** is a Python package to build predictive models using linear/logistic regression with a focus on performance and interpretation. It consists of several modules for data preprocessing, feature selection and model evaluation. The underlying methodology was developed at Python Predictions in the course of hundreds of business-related prediction challenges. It has been tweaked, tested and optimized over the years based on feedback from clients, our team, and academic researchers.

Main Features
=============

- Prepare a given pandas DataFrame for predictive modelling:

   - partition into train/selection/validation sets
   - create bins from continuous variables
   - regroup categorical variables based on statistical significance
   - replace missing values and
   - add columns with incidence rate per category/bin
 
- Perform univariate feature selection based on AUC
- Compute correlation matrix of predictors
- Find the suitable variables using forward feature selection
- Evaluate model performance and visualize the results

Getting started
===============

These instructions will get you a copy of the project up and running on your local machine for usage, development and testing purposes.

Requirements
------------

This package requires the usual Python packages for data science:

- numpy (>=1.19.4)
- pandas (>=1.1.5)
- scipy (>=1.5.4)
- scikit-learn (>=0.23.1)
- matplotlib (>=3.3.3)
- seaborn (>=0.11.0)


These packages, along with their versions are listed in ``requirements.txt`` and can be installed using ``pip``:    ::


  pip install -r requirements.txt


**Note**: if you want to install cobra with e.g. pip, you don't have to install all of these requirements as these are automatically installed with cobra itself.

Installation
------------

The easiest way to install cobra is using ``pip``   ::

  pip install -U pythonpredictions-cobra

Contributing to cobra
=====================

We'd love you to contribute to the development of cobra! There are many ways in which you can contribute, the most common of which is to contribute to the source code or documentation of the project. However, there are many other ways you can contribute (report issues, improve code coverage by adding unit tests, ...).
We use GitHub issue to track all bugs and feature requests. Feel free to open an issue in case you found a bug or in case you wish to see a new feature added.

For more details, check our `wiki <https://github.com/PythonPredictions/cobra/wiki/Contributing-guidelines-&-workflows>`_

Help and Support
================

Documentation
-------------

- HTML documentation of the `individual modules <https://pythonpredictions.github.io/cobra.io/docstring/modules.html>`_
- A step-by-step `tutorial <https://pythonpredictions.github.io/cobra.io/tutorial.html>`_

Outreach
-------------

- Check out the Data Science Leuven Meetup `talk <https://www.youtube.com/watch?v=w7ceZZqMEaA&feature=youtu.be>`_ by one of the core developers (second presentation)