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
  * Find the suitable variables using forward feature selection
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

__Note__: if you want to install cobra with e.g. pip, you don't have to install all of these requirements as these are automatically installed with cobra itself.

### Installation

As this package is an internal package that is not open-sourced, it is not available through `pip` or `conda`. As a result, the package has to be installed manually using the following steps:

  * Clone this repository.
  * Open a shell that can execute python code and navigate to the folder where this repo was cloned in.
  * Once you are in the folder, execute `python setup.py install` or `pip install .` (preferred).

### Usage

This section contains detailed examples for each step on how to use COBRA for building a predictive model. All classes and functions contain detailed documentation, so in case you want more information on a class or function, simply run the following python snippet:

```python
help(function_or_class_you_want_info_from)
```

In the examples below, we assume the data for model building is available in a pandas DataFrame called `basetable`. This DataFrame should contain an ID columns (e.g. customernumber), a target column (e.g. "TARGET") and a number of candidate predictors to build or model with.

```python
from cobra.preprocessing import PreProcessor

# Prepare data
# create instance of PreProcessor from parameters
# (many options possible, see source code for docs)
path = "path/to/store/preprocessing/pipeline/as/json/file/for/later/re-use/"
preprocessor = PreProcessor.from_params(serialization_path=path)

# split data into train-selection-validation set
# in the result, an additional column "split" will be created
# containing each of those values
basetable = preprocessor.train_selection_validation_split(
                basetable,
                target_column_name=target_column_name,
                train_prop=0.6, selection_prop=0.2,
                validation_prop=0.2)

# create list containing the column names of the discrete resp.
# continiuous variables
continuous_vars = []
discrete_vars = []

# fit the pipeline (will automatically be stored to "path" variable)
preprocessor.fit(basetable[basetable["split"]=="train"],
                 continuous_vars=continuous_vars,
                 discrete_vars=discrete_vars,
                 target_column_name=target_column_name)

# When you want to reuse the pipeline the next time, simply run
# preprocessor = PreProcessor.from_pipeline(path) and you're good to go!

# transform the data (e.g. perform discretisation, incidence replacement, ...)
basetable = preprocessor.transform(basetable,
                                   continuous_vars=continuous_vars,
                                   discrete_vars=discrete_vars)

```

Once the preprocessing pipeline is fitted and applied to your data, we are ready to start modelling. However, we could already compute the PIG tables here for later use:

```python
from cobra.evaluation import generate_pig_tables

pig_tables = generate_pig_tables(basetable[basetable["split"] == "selection"],
                                 id_column_name=id_column_name,
                                 target_column_name=target_column_name,
                                 preprocessed_predictors=preprocessed_predictors)
```

Once these PIG tables are computed, we can start with the _univariate preselection_:

```python
from cobra.model_building import univariate_selection
from cobra.evaluation import plot_univariate_predictor_quality
from cobra.evaluation import plot_correlation_matrix

# Get list of predictor names to use for univariate_selection
preprocessed_predictors = [col for col in basetable.columns if col.endswith("_enc")]

# perform univariate selection on preprocessed predictors:
df_auc = univariate_selection.compute_univariate_preselection(
    target_enc_train_data=basetable[basetable["split"] == "train"],
    target_enc_selection_data=basetable[basetable["split"] == "selection"],
    predictors=preprocessed_predictors,
    target_column=target_column_name,
    preselect_auc_threshold=0.53,  # if auc_selection <= 0.53 exclude predictor
    preselect_overtrain_threshold=0.05  # if (auc_train - auc_selection) >= 0.05 --> overfitting!
    )

# Plot df_auc to get a horizontal barplot:
plot_univariate_predictor_quality(df_auc)

# compute correlations between preprocessed predictors:
df_corr = (univariate_selection
           .compute_correlations(basetable[basetable["split"] == "train"],
                                 preprocessed_predictors))

# plot correlation matrix
plot_correlation_matrix(df_corr)

# get a list of predictors selection by the univariate selection
preselected_predictors = (univariate_selection
                          .get_preselected_predictors(df_auc))
```

After a preselection is done on the predictors, we can start the model building itself using forward feature selection to choose the right set of predictors:

```python
from cobra.model_building import ForwardFeatureSelection
from cobra.evaluation import plot_performance_curves
from cobra.evaluation import plot_variable_importance

forward_selection = ForwardFeatureSelection(max_predictors=30,
                                            pos_only=True)

# fit the forward feature selection on the train data
# has optional parameters to force and/or exclude certain predictors
forward_selection.fit(basetable[basetable["split"] == "train"],
                      target_column_name,
                      preselected_predictors)

# compute model performance (e.g. AUC for train-selection-validation)
performances = (forward_selection
                .compute_model_performances(basetable, target_column_name))

# plot performance curves
plot_performance_curves(performances)

# After plotting the performances and selecting the model,
# we can extract this model from the forward_selection class:
model = forward_selection.get_model_from_step(5)

# Note that chosen model has 6 variables (python lists start with index 0),
# which can be obtained as follows:
final_predictors = model.predictors
# We can also compute and plot the importance of each predictor in the model:
variable_importance = model.compute_variable_importance(
    basetable[basetable["split"] == "selection"]
)
plot_variable_importance(variable_importance)
```

Now that we have build and selected a final model, it is time to evaluate it against various evaluation metrics:

```python
from cobra.evaluation import Evaluator

# get numpy array of True target labels and predicted scores:
y_true = basetable[basetable["split"] == "selection"][target_column_name].values
y_pred = model.score_model(basetable[basetable["split"] == "selection"])

evaluator = Evaluator()
evaluator.fit(y_true, y_pred)  # Automatically find the best cut-off probability

# Get various scalar metrics such as accuracy, AUC, precision, recall, ...
evaluator.scalar_metrics

# Plot non-scalar evaluation metrics:
evaluator.plot_roc_curve()

evaluator.plot_confusion_matrix()

evaluator.plot_cumulative_gains()

evaluator.plot_lift_curve()

evaluator.plot_cumulative_response_curve()

```

## Development

We'd love you to contribute to the development of Cobra! To do so, clone the repo and create a _feature branch_ to do your development. Once your are finished, you can create a _pull request_ to merge it back into the main branch. Make sure to follow the _PEP 8_ styleguide if you make any changes to COBRA. You should also write or modify unit test for your changes if they are related to preprocessing!
