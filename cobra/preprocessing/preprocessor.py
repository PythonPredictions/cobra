"""
This module is a rework of the old cobra data_preparation.py. Here we will make
use of the classes for discretization, preprocessing of categorical variables
and incidence replacement. All of which will be employed to create a
preprocessing pipeline, which can be stored as a JSON file so that it can
easily be re-used for scoring.

Authors:

- Geert Verstraeten (methodology)
- Matthias Roels (implementation)
"""
# std lib imports
import json
from typing import Optional
import inspect
from datetime import datetime
import time

import logging
log = logging.getLogger(__name__)
# third party imports
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
# custom imports
from cobra.preprocessing import KBinsDiscretizer
from cobra.preprocessing import TargetEncoder
from cobra.preprocessing import CategoricalDataProcessor


class PreProcessor(BaseEstimator):

    """This class implements a so-called facade pattern to define a
    higher-level interface to work with the CategoricalDataProcessor,
    KBinsDiscretizer and TargetEncoder classes, so that their fit and transform
    methods are called in the correct order. Additionally, it provides features
    such as (de)serialization to/from JSON so that preprocessing pipelines can
    be stored and reloaded.

    Attributes
    ----------
    categorical_data_processor : CategoricalDataProcessor
        Instance of CategoricalDataProcessor to do the preprocessing of
        categorical variables
    discretizer : KBinsDiscretizer
        Instance of KBinsDiscretizer to do the prepocessing of continuous
        variables by means of discretization
    serialization_path : str
        path to save the pipeline to
    stratify_split : bool
        Whether or not to stratify the train-test split
    target_encoder : TargetEncoder
        Instance of TargetEncoder to do the incidence replacement
    """

    def __init__(self, categorical_data_processor: CategoricalDataProcessor,
                 discretizer: KBinsDiscretizer,
                 target_encoder: TargetEncoder,
                 serialization_path: str=None,
                 is_fitted: bool=False):

        self.serialization_path = serialization_path

        self._categorical_data_processor = categorical_data_processor
        self._discretizer = discretizer
        self._target_encoder = target_encoder

        self._is_fitted = is_fitted

    @classmethod
    def from_params(cls,
                    n_bins: int=10,
                    strategy: str="quantile",
                    closed: str="right",
                    auto_adapt_bins: bool=False,
                    starting_precision: int=0,
                    label_format: str="{} - {}",
                    change_endpoint_format: bool=False,
                    regroup: bool=True,
                    regroup_name: str="Other",
                    keep_missing: bool=True,
                    category_size_threshold: int=5,
                    p_value_threshold: float=0.001,
                    scale_contingency_table: bool=True,
                    forced_categories: dict={},
                    weight: float=0.0,
                    imputation_strategy: str="mean",
                    serialization_path: Optional[str]=None):
        """Constructor to instantiate PreProcessor from all the parameters
        that can be set in all its required (attribute) classes
        along with good default values.

        Parameters
        ----------
        n_bins : int, optional
            Number of bins to produce. Raises ValueError if ``n_bins < 2``.
        strategy : str, optional
            Binning strategy. Currently only ``uniform`` and ``quantile``
            e.g. equifrequency is supported
        closed : str, optional
            Whether to close the bins (intervals) from the left or right
        auto_adapt_bins : bool, optional
            reduces the number of bins (starting from n_bins) as a function of
            the number of missings
        starting_precision : int, optional
            Initial precision for the bin edges to start from,
            can also be negative. Given a list of bin edges, the class will
            automatically choose the minimal precision required to have proper
            bins e.g. ``[5.5555, 5.5744, ...]`` will be rounded
            to ``[5.56, 5.57, ...]``. In case of a negative number, an attempt
            will be made to round up the numbers of the bin edges
            e.g. ``5.55 -> 10``, ``146 -> 100``, ...
        label_format : str, optional
            format string to display the bin labels
            e.g. ``min - max``, ``(min, max]``, ...
        change_endpoint_format : bool, optional
            Whether or not to change the format of the lower and upper bins
            into ``< x`` and ``> y`` resp.
        regroup : bool
            Whether or not to regroup categories
        regroup_name : str
            New name of the non-significant regrouped variables
        keep_missing : bool
            Whether or not to keep missing as a separate category
        category_size_threshold : int
            minimal size of a category to keep it as a separate category
        p_value_threshold : float
            Significance threshold for regrouping.
        forced_categories : dict
            Map to prevent certain categories from being group into ``Other``
            for each column - dict of the form ``{col:[forced vars]}``.
        scale_contingency_table : bool
            Whether contingency table should be scaled before chi^2.'
        weight : float, optional
            Smoothing parameters (non-negative). The higher the value of the
            parameter, the bigger the contribution of the overall mean.
            When set to zero, there is no smoothing
            (e.g. the pure target incidence is used).
        imputation_strategy : str, optional
            in case there is a particular column which contains new categories,
            the encoding will lead to NULL values which should be imputed.
            Valid strategies are to replace with the global mean of the train
            set or the min (resp. max) incidence of the categories of that
            particular variable.
        serialization_path : str, optional
            path to save the pipeline to

        Returns
        -------
        PreProcessor
            Description
        """
        categorical_data_processor = CategoricalDataProcessor(
            regroup,
            regroup_name,
            keep_missing,
            category_size_threshold,
            p_value_threshold,
            scale_contingency_table,
            forced_categories)
        discretizer = KBinsDiscretizer(n_bins, strategy, closed,
                                       auto_adapt_bins,
                                       starting_precision,
                                       label_format,
                                       change_endpoint_format)

        target_encoder = TargetEncoder(weight)

        return cls(categorical_data_processor, discretizer, target_encoder,
                   serialization_path)

    @classmethod
    def from_pipeline(cls, pipeline_path: str):
        """Constructor to instantiate PreProcessor from a (fitted) pipeline,
        stored as a JSON file.

        Parameters
        ----------
        pipeline_path : str
            path to the (fitted) pipeline

        Returns
        -------
        PreProcessor
            Instance of PreProcessor instantiated from a stored pipeline

        Raises
        ------
        ValueError
            Description
        """
        with open(pipeline_path, "r") as file:
            pipeline = json.load(file)

        if not PreProcessor._is_valid_pipeline(pipeline):
            raise ValueError("Invalid pipeline")  # To do: specify error

        categorical_data_processor = CategoricalDataProcessor()
        categorical_data_processor.set_attributes_from_dict(
            pipeline["categorical_data_processor"]
        )

        discretizer = KBinsDiscretizer()
        discretizer.set_attributes_from_dict(pipeline["discretizer"])

        target_encoder = TargetEncoder()
        target_encoder.set_attributes_from_dict(pipeline["target_encoder"])

        return cls(categorical_data_processor, discretizer, target_encoder,
                   is_fitted=pipeline["_is_fitted"])

    def fit(self, train_data: pd.DataFrame, continuous_vars: list,
            discrete_vars: list, target_column_name: str):
        """Fit the data to the preprocessing pipeline

        Parameters
        ----------
        train_data : pd.DataFrame
            Data to be preprocessed
        continuous_vars : list
            list of continuous variables
        discrete_vars : list
            list of discrete variables
        target_column_name : str
            Column name of the target
        """

        # get list of all variables
        preprocessed_variable_names = (PreProcessor
                                       ._get_variable_list(continuous_vars,
                                                           discrete_vars))

        log.info("Starting to fit pipeline")
        start = time.time()

        # Fit discretizer, categorical preprocessor & target encoder
        # Note that in order to fit target_encoder, we first have to transform
        # the data using the fitted discretizer & categorical_data_processor
        if continuous_vars:
            begin = time.time()
            self._discretizer.fit(train_data, continuous_vars)
            log.info("Fitting KBinsDiscretizer took {} seconds"
                     .format(time.time() - begin))

            train_data = self._discretizer.transform(train_data,
                                                     continuous_vars)
        if discrete_vars:
            begin = time.time()
            self._categorical_data_processor.fit(train_data,
                                                 discrete_vars,
                                                 target_column_name)
            log.info("Fitting categorical_data_processor class took {} seconds"
                     .format(time.time() - begin))

            train_data = (self._categorical_data_processor
                          .transform(train_data, discrete_vars))

        begin = time.time()
        self._target_encoder.fit(train_data, preprocessed_variable_names,
                                 target_column_name)
        log.info("Fitting TargetEncoder took {} seconds"
                 .format(time.time() - begin))

        self._is_fitted = True  # set fitted boolean to True
        # serialize the pipeline to store the fitted output along with the
        # various parameters that were used
        self._serialize()

        log.info("Fitting and serializing pipeline took {} seconds"
                 .format(time.time() - start))

    def transform(self, data: pd.DataFrame, continuous_vars: list,
                  discrete_vars: list) -> pd.DataFrame:
        """Transform the data by applying the preprocessing pipeline

        Parameters
        ----------
        data : pd.DataFrame
            Data to be preprocessed
        continuous_vars : list
            list of continuous variables
        discrete_vars : list
            list of discrete variables

        Returns
        -------
        pd.DataFrame
            Transformed (preprocessed) data

        Raises
        ------
        NotFittedError
            In case PreProcessor was not fitted first
        """

        start = time.time()

        if not self._is_fitted:
            msg = ("This {} instance is not fitted yet. Call 'fit' with "
                   "appropriate arguments before using this method.")

            raise NotFittedError(msg.format(self.__class__.__name__))

        preprocessed_variable_names = (PreProcessor
                                       ._get_variable_list(continuous_vars,
                                                           discrete_vars))

        if continuous_vars:
            data = self._discretizer.transform(data, continuous_vars)

        if discrete_vars:
            data = self._categorical_data_processor.transform(data,
                                                              discrete_vars)

        data = self._target_encoder.transform(data,
                                              preprocessed_variable_names)

        log.info("Transforming data took {} seconds"
                 .format(time.time() - start))

        return data

    def fit_transform(self, train_data: pd.DataFrame, continuous_vars: list,
                      discrete_vars: list,
                      target_column_name: str) -> pd.DataFrame:
        """Fit preprocessing pipeline and transform the data

        Parameters
        ----------
        train_data : pd.DataFrame
            Data to be preprocessed
        continuous_vars : list
            list of continuous variables
        discrete_vars : list
            list of discrete variables
        target_column_name : str
            Column name of the target

        Returns
        -------
        pd.DataFrame
            Transformed (preprocessed) data
        """

        self.fit(train_data, continuous_vars, discrete_vars,
                 target_column_name)

        return self.transform(train_data, continuous_vars, discrete_vars)

    @staticmethod
    def train_selection_validation_split(data: pd.DataFrame,
                                         target_column_name: str,
                                         train_prop: float=0.6,
                                         selection_prop: float=0.2,
                                         validation_prop: float=0.2,
                                         stratify_split=True)->pd.DataFrame:
        """Split dataset into train-selection-validation datasets and merge
        them into one big DataFrame with an additional column "split"
        indicating to which dataset the corresponding row belongs to.

        Parameters
        ----------
        data : pd.DataFrame
            Input dataset to split into train-selection and validation sets
        target_column_name : str
            Name of the target column
        train_prop : float, optional
            Percentage data to put in train set
        selection_prop : float, optional
            Percentage data to put in selection set
        validation_prop : float, optional
            Percentage data to put in validation set
        stratify_split : bool, optional
            Whether or not to stratify the train-test split

        Returns
        -------
        pd.DataFrame
            DataFrame with additional split column
        """

        if train_prop + selection_prop + validation_prop != 1.0:
            raise ValueError("The sum of train_prop, selection_prop and "
                             "validation_prop cannot differ from 1.0")

        if selection_prop == 0.0:
            raise ValueError("selection_prop cannot be zero!")

        column_names = list(data.columns)

        predictors = [col for col in column_names if col != target_column_name]

        # for the first split, take sum of selection & validation pct as
        # test pct
        test_prop = selection_prop + validation_prop
        # To further split our test set into selection + validation set,
        # we have to modify validation pct because we only have test_prop of
        # the data available anymore for further splitting!
        validation_prop_modif = validation_prop / test_prop

        X = data[predictors]
        y = data[target_column_name]

        stratify = None
        if stratify_split:
            stratify = y

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_prop,
            random_state=42,
            stratify=stratify
            )

        df_train = pd.DataFrame(X_train, columns=predictors)
        df_train[target_column_name] = y_train
        df_train["split"] = "train"

        # If there is no validation percentage, return train-selection sets
        # only
        if validation_prop == 0.0:
            df_selection = pd.DataFrame(X_test, columns=predictors)
            df_selection[target_column_name] = y_test
            df_selection["split"] = "selection"

            return (pd.concat([df_train, df_selection])
                    .reset_index(drop=True))

        if stratify_split:
            stratify = y_test

        X_sel, X_val, y_sel, y_val = train_test_split(
            X_test, y_test,
            test_size=validation_prop_modif,
            random_state=42,
            stratify=stratify
            )

        df_selection = pd.DataFrame(X_sel, columns=predictors)
        df_selection[target_column_name] = y_sel
        df_selection["split"] = "selection"

        df_validation = pd.DataFrame(X_val, columns=predictors)
        df_validation[target_column_name] = y_val
        df_validation["split"] = "validation"

        return (pd.concat([df_train, df_selection, df_validation])
                .reset_index(drop=True))

    def _serialize(self) -> dict:
        """Serialize the preprocessing pipeline by writing all its required
        parameters to a JSON file.

        Returns
        -------
        dict
            Return the pipeline as a dictionary
        """
        pipeline = {
            "metadata": {
                "timestamp": datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            }
        }

        pipeline["categorical_data_processor"] = (self
                                                  ._categorical_data_processor
                                                  .attributes_to_dict())

        pipeline["discretizer"] = self._discretizer.attributes_to_dict()
        pipeline["target_encoder"] = (self._target_encoder
                                      .attributes_to_dict())

        pipeline["_is_fitted"] = True

        if self.serialization_path:
            path = self.serialization_path
        else:
            path = "./pipeline_tmp.json"

        with open(path, "w") as file:
            json.dump(pipeline, file)

        return pipeline

    @staticmethod
    def _is_valid_pipeline(pipeline: dict) -> bool:
        """Validate the loaded pipeline by checking if all required parameters
        are present (and no others!).

        Parameters
        ----------
        pipeline : dict
            Loaded pipeline from json file
        """
        keys = inspect.getfullargspec(PreProcessor.from_params).args
        valid_keys = set([key for key in keys
                          if key not in ["cls", "serialization_path"]])

        input_keys = set()
        for key in pipeline:
            if key in ["categorical_data_processor", "discretizer",
                       "target_encoder"]:
                input_keys = input_keys.union(set(pipeline[key].keys()))
            elif key != "metadata":
                input_keys.add(key)

        input_keys = sorted(list(input_keys))
        input_keys = [key for key in input_keys if not key.startswith("_")]

        return sorted(list(valid_keys)) == sorted(list(input_keys))

    @staticmethod
    def _get_variable_list(continuous_vars: list, discrete_vars: list) -> list:
        """merge lists of continuous_vars and discrete_vars and add suffix
        "_bin" resp. "_processed" to the predictors

        Parameters
        ----------
        continuous_vars : list
            list of continuous variables
        discrete_vars : list
            list of discrete variables

        Returns
        -------
        list
            Merged list of predictors with proper suffixes added

        Raises
        ------
        ValueError
            in case both lists are empty
        """
        var_list = ([col + "_processed" for col in discrete_vars]
                    + [col + "_bin" for col in continuous_vars])

        if not var_list:
            raise ValueError("Variable var_list is None or empty list")

        return var_list
