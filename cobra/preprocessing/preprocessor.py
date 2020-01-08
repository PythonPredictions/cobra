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

import cobra.utils as utils


class PreProcessor(BaseEstimator):

    """Summary

    Attributes
    ----------
    categorical_data_processor : CategoricalDataProcessor
        Instance of CategoricalDataProcessor to do the prepocessing of
        categorical variables
    discretizer : KBinsDiscretizer
        Instance of KBinsDiscretizer to do the prepocessing of continuous
        variables by means of discretization
    numeric_threshold : int
        Threshold to decide whether a numeric variable is in fact a categorical
        one based on the number of unique values of that variable
    selection_pct : float
        Percentage of data to add to selection dataset
    serialization_path : str
        path to save the pipeline to
    stratify_split : bool
        Whether or not to stratify the train-test split
    target_encoder : TargetEncoder
        Instance of TargetEncoder to do the incidence replacement
    train_pct : float
        Percentage of data to add to training dataset
    validation_pct : float
        Percentage of data to add to validation dataset
    """

    def __init__(self,
                 train_pct: float,
                 selection_pct: float,
                 validation_pct: float,
                 stratify_split: bool,
                 categorical_data_processor: CategoricalDataProcessor,
                 discretizer: KBinsDiscretizer,
                 target_encoder: TargetEncoder,
                 threshold_numeric_is_categorical: int=None,
                 serialization_path: str=None,
                 continuous_vars: list=[],
                 discrete_vars: list=[]):

        self.train_pct = train_pct
        self.selection_pct = selection_pct
        self.validation_pct = validation_pct
        self.stratify_split = stratify_split

        self.numeric_threshold = threshold_numeric_is_categorical

        self.serialization_path = serialization_path

        self.categorical_data_processor = categorical_data_processor
        self.discretizer = discretizer
        self.target_encoder = target_encoder

        # placeholders for columns by datatype
        # Included as constructor argument to get them from pipeline
        self._is_fitted = False
        self._continuous_vars = continuous_vars
        self._discrete_vars = discrete_vars

    @classmethod
    def from_params(cls,
                    train_pct: float,
                    selection_pct: float,
                    validation_pct: float,
                    stratify_split: bool=True,
                    threshold_numeric_is_categorical: int=None,
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
                    serialization_path: Optional[str]=None):
        """Constructor to instantiate PreProcessor from all the parameters
        that can be set in all its required classes.

        Parameters
        ----------
        train_pct : float
            Percentage of data to add to training dataset
        selection_pct : float
            Percentage of data to add to selection dataset
        validation_pct : float
            Percentage of data to add to validation dataset
        stratify_split : bool, optional
            Whether or not to stratify the train-test split
        threshold_numeric_is_categorical : int, optional
            Threshold to decide whether a numeric variable is in fact a
            categorical one based on the number of unique values of
            that variable
        n_bins : int, optional
            Number of bins to produce. Raises ValueError if ``n_bins < 2``.
        strategy : str, optional
            Binning strategy. Currently only "uniform" and "quantile"
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
            bins e.g. [5.5555, 5.5744, ...] will be rounded
            to [5.56, 5.57, ...]. In case of a negative number, an attempt will
            be made to round up the numbers of the bin edges
            e.g. 5.55 -> 10, 146 -> 100, ...
        label_format : str, optional
            format string to display the bin labels
            e.g. min - max, (min, max], ...
        change_endpoint_format : bool, optional
            Whether or not to change the format of the lower and upper bins
            into "< x" and "> y" resp.
        regroup : bool
            Whether or not to regroup categories
        regroup_name : str
            New name of the non-significant regrouped variables
        keep_missing : bool
            Whether or not to keep missing as a separate category
        category_size_threshold : int
            minimal size of a category to keep it as a separate category
        p_value_threshold : float
            Significance threshold for regroupping.
        forced_categories : dict
            Map to prevent certain categories from being group into "Other"
            for each colum - dict of the form {col:[forced vars]}.
        scale_contingency_table : bool
            Whether contingency table should be scaled before chi^2.'
        weight : float, optional
            Smoothing parameters (non-negative). The higher the value of the
            parameter, the bigger the contribution of the overall mean.
            When set to zero, there is no smoothing
            (e.g. the pure target incidence is used).
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

        if not threshold_numeric_is_categorical:
            threshold_numeric_is_categorical = n_bins

        return cls(train_pct, selection_pct, validation_pct, stratify_split,
                   categorical_data_processor, discretizer, target_encoder,
                   threshold_numeric_is_categorical, serialization_path)

    @classmethod
    def from_pipeline(cls, pipeline_path: str):
        """Summary

        Parameters
        ----------
        pipeline_path : str
            Description

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

        return cls(pipeline["train_pct"], pipeline["selection_pct"],
                   pipeline["validation_pct"], pipeline["stratify_split"],
                   categorical_data_processor,
                   discretizer, target_encoder,
                   pipeline["threshold_numeric_is_categorical"],
                   continuous_vars=pipeline["continuous_vars"],
                   discrete_vars=pipeline["discrete_vars"])

    def fit(self, data: pd.DataFrame, target_column_name: str,
            id_column_name: str=None):
        """Fit the data to the preprocessing pipeline

        Parameters
        ----------
        data : pd.DataFrame
            Data to be preprocessed
        target_column_name : str
            Name of the target column
        id_column_name : str, optional
            Name of the id column
        """
        # get list of all variables
        var_list = self._get_variable_list(data, target_column_name,
                                           id_column_name)

        data = (PreProcessor
                .train_selection_validation_split(data, target_column_name,
                                                  self.train_pct,
                                                  self.selection_pct,
                                                  self.validation_pct,
                                                  self.stratify_split))

        # Select train data to fit preprocessing pipeline
        train_data = data[data["split"] == "train"]

        # Fit discretizer, categorical preprocessor & target encoder
        # Note that in order to fit target_encoder, we first have to transform
        # the data using the fitted discretizer & categorical_data_processor
        if self._continuous_vars:
            self.discretizer.fit(train_data, self._continuous_vars)
            train_data = self.discretizer.transform(train_data,
                                                    self._continuous_vars)

        if self._discrete_vars:
            self.categorical_data_processor.fit(train_data,
                                                self._discrete_vars,
                                                target_column_name)
            train_data = (self.categorical_data_processor
                          .transform(train_data, self._discrete_vars))

        self.target_encoder.fit(train_data, var_list, target_column_name)

        self._is_fitted = True  # set fitted boolean to True
        # serialize the pipeline to store the fitted output along with the
        # various parameters that were used
        self._serialize()

    def transform(self, data: pd.DataFrame, target_column_name: str,
                  id_column_name: str=None) -> pd.DataFrame:
        """Summary

        Parameters
        ----------
        data : pd.DataFrame
            Description
        target_column_name : str
            Description
        id_column_name : str, optional
            Description

        Returns
        -------
        pd.DataFrame
            Description
        """

        if not self._is_fitted:
            msg = ("This {} instance is not fitted yet. Call 'fit' with "
                   "appropriate arguments before using this method.")

            raise NotFittedError(msg.format(self.__class__.__name__))

        var_lists = ([col + "_processed" for col in self._discrete_vars]
                     + [col + "_bin" for col in self._continuous_vars])

        data = self.discretizer.transform(data, self._continuous_vars)
        data = self.categorical_data_processor.transform(data,
                                                         self._discrete_vars)

        data = self.target_encoder.transform(data, var_lists)

        return data

    @staticmethod
    def train_selection_validation_split(data: pd.DataFrame,
                                         target_column_name: str,
                                         train_pct: float=0.6,
                                         selection_pct: float=0.2,
                                         validation_pct: float=0.2,
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
        train_pct : float, optional
            Percentage data to put in train set
        selection_pct : float, optional
            Percentage data to put in selection set
        validation_pct : float, optional
            Percentage data to put in validation set
        stratify_split : bool, optional
            Whether or not to stratify the train-test split

        Returns
        -------
        pd.DataFrame
            Description
        """
        column_names = list(data.columns)

        predictors = [col for col in column_names if col != target_column_name]

        test_pct = selection_pct + validation_pct

        X = data[predictors]
        y = data[target_column_name]

        stratify = None
        if stratify_split:
            stratify = y

        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=test_pct,
                                                            random_state=42,
                                                            stratify=stratify)

        if stratify_split:
            stratify = y_test

        X_sel, X_val, y_sel, y_val = train_test_split(X_test, y_test,
                                                      test_size=validation_pct,
                                                      random_state=42,
                                                      stratify=stratify)

        df_train = pd.DataFrame(X_train, columns=predictors)
        df_train[target_column_name] = y_train
        df_train["split"] = "train"

        df_selection = pd.DataFrame(X_sel, columns=predictors)
        df_selection[target_column_name] = y_sel
        df_selection["split"] = "selection"

        df_validation = pd.DataFrame(X_val, columns=predictors)
        df_validation[target_column_name] = y_val
        df_validation["split"] = "validation"

        return (pd.concat([df_train, df_selection, df_validation])
                .reset_index(drop=True))

    def _get_variable_list(self, data: pd.DataFrame,
                           target_column_name: str,
                           id_column_name: str=None):
        """Get list of variables and split into numeric and categorical

        Parameters
        ----------
        data : pd.DataFrame
            Data to be preprocessed
        target_column_name : str
            Name of the target column
        id_column_name : str, optional
            Name of the id column
        """

        columns_by_datatype = utils.get_column_datatypes(data,
                                                         target_column_name,
                                                         id_column_name,
                                                         self.numeric_threshold
                                                         )

        self._continuous_vars = columns_by_datatype["numeric_variables"]
        self._discrete_vars = columns_by_datatype["categorical_variables"]

        log.info("Numeric variables: {}".format(self._continuous_vars))
        log.info("Categorical variables:".format(self._discrete_vars))

        var_list = ([col + "_processed" for col in self._discrete_vars]
                    + [col + "_bin" for col in self._continuous_vars])

        if not var_list:
            raise ValueError("Variable var_list is None or empty list")

        return var_list

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

        pipeline["train_pct"] = self.train_pct
        pipeline["selection_pct"] = self.selection_pct
        pipeline["validation_pct"] = self.validation_pct
        pipeline["stratify_split"] = self.stratify_split
        pipeline["threshold_numeric_is_categorical"] = self.numeric_threshold

        pipeline["categorical_data_processor"] = (self
                                                  .categorical_data_processor
                                                  .attributes_to_dict())

        pipeline["discretizer"] = self.discretizer.attributes_to_dict()
        pipeline["target_encoder"] = (self.target_encoder.attributes_to_dict())

        pipeline["continuous_vars"] = self._continuous_vars
        pipeline["discrete_vars"] = self._discrete_vars

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

        return valid_keys == input_keys
