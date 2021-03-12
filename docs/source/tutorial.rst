Tutorial
========

This section we will walk you through all the required steps to build a predictive model using cobra. All classes and functions used here are well-documented. In case you want more information on a class or function, simply read the corresponding parts in the documentation or run the following python snippet from e.g. a notebook:

.. code-block:: python

    help(function_or_class_you_want_info_from)

Building a good model involves three steps

    - preprocessing: properly prepare the predictors (a synonym for "feature" or variable that we use throughout this tutorial) for modelling.
    - feature selection: automatically select a subset of predictors which contribute most to the target variable or output in which you are interested.
    - model evaluation: once a model has been build, a detailed evaluation can be performed by computing all sorts of evaluation metrics.

In the examples below, we assume the data for model building is available in a pandas DataFrame called ``basetable``. This DataFrame should at least contain an ID column (e.g. "customernumber"), a target column (e.g. "TARGET") and a number of candidate predictors (features) to build a model with.

Preprocessing
-------------

The first part focusses on preparing the predictors for modelling by:

- Splitting the dataset into training, selection and validation datasets.
- binning continuous variables into discrete intervals
- Replacing missing values of both categorical and continuous variables (which are now binned) with an additional "Missing" bin/category
- Regrouping categories in new category "other"
- Replacing bins/categories with their corresponding incidence rate per category/bin.

This will be taken care of by the ``PreProcessor`` class, which has a scikit-learn like interface (i.e. ``fit`` & ``transform``)

.. code-block:: python

    import json
    from cobra.preprocessing import PreProcessor

    # Prepare data
    # create instance of PreProcessor from parameters
    # There are many options possible, see API reference, but here
    # we will use all the defaults
    preprocessor = PreProcessor.from_params()

    # split data into train-selection-validation set
    # in the result, an additional column "split" will be created
    # containing each of those values
    basetable = preprocessor.train_selection_validation_split(
                    basetable,
                    target_column_name=target_column_name,
                    train_prop=0.6, selection_prop=0.2,
                    validation_prop=0.2)

    # create list containing the column names of the discrete resp.
    # continuous variables
    continuous_vars = []
    discrete_vars = []

    # fit the pipeline
    preprocessor.fit(basetable[basetable["split"]=="train"],
                     continuous_vars=continuous_vars,
                     discrete_vars=discrete_vars,
                     target_column_name=target_column_name)

    # store fitted preprocessing pipeline as a JSON file
    pipeline = preprocessor.serialize_pipeline()

    # I/O outside of PreProcessor to allow flexibility (e.g. upload to S3, ...)
    path = "path/to/store/preprocessing/pipeline/as/json/file/for/later/re-use.json"
    with open(path, "w") as file:
        json.dump(pipeline, file)

    # transform the data (e.g. perform discretisation, incidence replacement, ...)
    basetable = preprocessor.transform(basetable,
                                       continuous_vars=continuous_vars,
                                       discrete_vars=discrete_vars)

    # When you want to reuse the pipeline the next time, simply load it back in again
    # using the following snippet:
    # with open(path, "r") as file:
    #     pipeline = json.load(file)
    # preprocessor = PreProcessor.from_pipeline(pipeline) and you're good to go!

Feature selection
-----------------

Once the predictors are properly prepared, we can start building a predictive model, which boils down to selecting the right predictors from the dataset to train a model on. As a dataset typically contains many predictors, we can first perform a univariate preselection to rule out any predictor with little to no predictive power.

This preselection is based on an AUC threshold of a univariate model on the train and selection datasets. As the AUC just calculates the quality of a ranking, all monotonous transformations of a given ranking (i.e. transformations that do not alter the ranking itself) will lead to the same AUC. Hence, pushing a categorical variable (incl. a binned continuous variable) through a logistic regression will produce exactly the same ranking as using target encoding, as it will produce the exact same output: a ranking of the categories on the training/selection set. Therefore, no univariate model is trained here as the target encoded train and selection data is used as predicted scores to compute the AUC with against the target.

.. code-block:: python

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

After an initial preselection on the predictors, we can start building the model itself using forward feature selection to choose the right set of predictors. Since we use target encoding on all our predictors, we will only consider models with positive coefficients (no sign flip should occur) as this makes the model more interpretable.

.. code-block:: python

    from cobra.model_building import ForwardFeatureSelection
    from cobra.evaluation import plot_performance_curves
    from cobra.evaluation import plot_variable_importance

    forward_selection = ForwardFeatureSelection(max_predictors=30,
                                                pos_only=True)

    # fit the forward feature selection on the train data
    # has optional parameters to force and/or exclude certain predictors (see docs)
    forward_selection.fit(basetable[basetable["split"] == "train"],
                          target_column_name,
                          preselected_predictors)

    # compute model performance (e.g. AUC for train-selection-validation)
    performances = (forward_selection
                    .compute_model_performances(basetable, target_column_name))

    # plot performance curves
    plot_performance_curves(performances)

Based on the performance curves (AUC per model with a particular number of predictors in case of logistic regression), a final model can then be chosen and the variables importance can be plotted:

.. code-block:: python

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

**Note**: variable importance is based on correlation of the predictor with the *model scores* (and not the true labels!).

Finally, we can again export the model to a dictionary to store it as JSON

.. code-block:: python

    model_dict = model.serialize()

    with open(path, "w") as file:
        json.dump(model_dict, file)

    # To reload the model again from a JSON file, run the following snippet:
    # from cobra.model_building import LogisticRegressionModel
    # with open(path, "r") as file:
    #     model_dict = json.load(file)
    # model = LogisticRegressionModel()
    # model.deserialize(model_dict)

Evaluation
----------

Now that we have build and selected a final model, it is time to evaluate it against various evaluation metrics:

.. code-block:: python

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

Additionally, we can also compute the output needed to plot the so-called Predictor Insights Graphs (PIGs in short). These are graphs that represents the insights of the relationship between a single predictor (e.g. age) and the target (e.g. burnouts). This is a graph where the predictor is binned into groups, and where we represent group size in bars and group (target) incidence in a colored line. We have the option to force order of predictor values.

.. code-block:: python

    from cobra.evaluation import generate_pig_tables
    from cobra.evaluation import plot_incidence

    predictor_list = [col for col in basetable.columns
                      if col.endswith("_bin") or col.endswith("_processed")]
    pig_tables = generate_pig_tables(basetable[basetable["split"] == "selection"],
                                     id_column_name=id_column_name,
                                     target_column_name=target_column_name,
                                     preprocessed_predictors=predictor_list)
    # Plot PIGs
    plot_incidence(pig_tables, 'predictor_name', predictor_order)                                     