"""
Incidence Replacement Module. The implementation is inspired by
https://contrib.scikit-learn.org/categorical-encoding/index.html

Authors:

- Geert Verstraeten (methodology)
- Matthias Roels (implementation)
"""
import logging
log = logging.getLogger(__name__)

#import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError


class TargetEncoder(BaseEstimator):

    """Target encoding for categorical features.

    Replace each value of the categorical feature with the average of the
    target values (in case of a binary target, this is the incidence of the
    group). This encoding scheme is also called Mean encoding.

    The main problem with Target encoding is overfitting; the fact that we are
    encoding the feature based on target classes may lead to data leakage,
    rendering the feature biased. This can be solved using some type of
    regularization. A popular way to handle this is to use cross-validation
    and compute the means in each out-of-fold. However, the approach
    implemented here makes use of additive smoothing
    (https://en.wikipedia.org/wiki/Additive_smoothing)

    Attributes
    ----------
    imputation_strategy : str
        in case there is a particular column which contains new categories,
        the encoding will lead to NULL values which should be imputed.
        Valid strategies are to replace with the global mean of the train
        set or the min (resp. max) incidence of the categories of that
        particular variable.
    weight : float
        Smoothing parameters (non-negative). The higher the value of the
        parameter, the bigger the contribution of the overall mean. When set to
        zero, there is no smoothing (e.g. the pure target incidence is used).
    """

    valid_strategies = ("mean", "min", "max")

    def __init__(self, weight: float=0.0,
                 imputation_strategy: str="mean"):

        if weight < 0:
            raise ValueError("The value of weight cannot be smaller than zero")
        elif imputation_strategy not in self.valid_strategies:
            raise ValueError("Valid options for 'imputation_strategy' are {}."
                             " Got imputation_strategy={!r} instead"
                             .format(self.valid_strategies,
                                     imputation_strategy))

        self.weight = weight
        self.imputation_strategy = imputation_strategy

        self._mapping = {}  # placeholder for fitted output
        # placeholder for the global incidence of the data used for fitting
        self._global_mean = None

    def attributes_to_dict(self) -> dict:
        """Return the attributes of TargetEncoder in a dictionary

        Returns
        -------
        dict
            Contains the attributes of TargetEncoder instance with the names
            as keys
        """
        params = self.get_params()

        params["_mapping"] = {
            key: value.to_dict()
            for key, value in self._mapping.items()
        }

        params["_global_mean"] = self._global_mean

        return params

    def set_attributes_from_dict(self, params: dict):
        """Set instance attributes from a dictionary of values with key the
        name of the attribute.

        Parameters
        ----------
        params : dict
            Contains the attributes of TargetEncoder with their
            names as key.
        """

        if "weight" in params and type(params["weight"]) == float:
            self.weight = params["weight"]

        if ("imputation_strategy" in params and
                params["imputation_strategy"] in self.valid_strategies):

            self.imputation_strategy = params["imputation_strategy"]

        if "_global_mean" in params and type(params["_global_mean"]) == float:
            self._global_mean = params["_global_mean"]

        _mapping = {}
        if "_mapping" in params and type(params["_mapping"]) == dict:
            _mapping = params["_mapping"]

        def dict_to_series(key, value):
            s = pd.Series(value)
            s.index.name = key
            return s

        self._mapping = {
            key: dict_to_series(key, value)
            for key, value in _mapping.items()
        }

        return self

    def fit(self, data: pd.DataFrame, column_names: list,
            target_column: str):
        """Fit the TargetEncoder to the data

        Parameters
        ----------
        data : pd.DataFrame
            data used to compute the mapping to encode the categorical
            variables with.
        column_names : list
            Columns of data to be encoded
        target_column : str
            Column name of the target
        """

        # compute global mean (target incidence in case of binary target)
        y = data[target_column]
        self._global_mean = y.sum() / y.count()

        for column in column_names:
            if column not in data.columns:
                log.warning("DataFrame has no column '{}', so it will be "
                            "skipped in fitting" .format(column))
                continue

            self._mapping[column] = self._fit_column(data[column], y)

    def _fit_column(self, X: pd.Series, y: pd.Series) -> pd.Series:
        """Summary

        Parameters
        ----------
        X : pd.Series
            data used to compute the encoding mapping for an individual
            categorical variable.
        y : pd.Series
            series containing the targets for each observation

        Returns
        -------
        pd.Series
            Mapping containing the value to replace each group of the
            categorical with.
        """
        stats = y.groupby(X).agg(["mean", "count"])

        # Note if self.weight = 0, we have the ordinary incidence replacement
        numerator = (stats["count"]*stats["mean"]
                     + self.weight * self._global_mean)

        denominator = stats["count"] + self.weight

        return numerator/denominator

    def transform(self, data: pd.DataFrame,
                  column_names: list) -> pd.DataFrame:
        """Replace (e.g. encode) categories of each column with its average
        incidence which was computed when the fit method was called

        Parameters
        ----------
        X : pd.DataFrame
            data to encode
        column_names : list
             Columns of data to be encoded

        Returns
        -------
        pd.DataFrame
            transformed data

        Raises
        ------
        NotFittedError
            Exception when TargetEncoder was not fitted before calling this
            method

        """
        if (len(self._mapping) == 0) or (self._global_mean is None):
            msg = ("This {} instance is not fitted yet. Call 'fit' with "
                   "appropriate arguments before using this method.")

            raise NotFittedError(msg.format(self.__class__.__name__))

        for column in column_names:

            if column not in data.columns:
                log.warning("Unknown column '{}' will be skipped"
                            .format(column))
                continue
            elif column not in self._mapping:
                log.warning("Column '{}' is not in fitted output "
                            "and will be skipped".format(column))
                continue

            data = self._transform_column(data, column)

        return data

    def _transform_column(self, data: pd.DataFrame,
                          column_name: str) -> pd.DataFrame:
        """Replace (e.g. encode) categories of each column with its average
        incidence which was computed when the fit method was called

        Parameters
        ----------
        X : pd.DataFrame
            data to encode
        column_name : str
            Name of the column in data to be encoded

        Returns
        -------
        pd.DataFrame
            transformed data
        """
        new_column = TargetEncoder._clean_column_name(column_name)

        # Convert dtype to float because when the original dtype
        # is of type "category", the resulting dtype is also of type
        # "category"
        data[new_column] = (data[column_name].map(self._mapping[column_name])
                            .astype("float"))

        # In case of categorical data, it could be that new categories will
        # emerge which were not present in the train set, so this will result
        # in missing values (which should be replaced)
        if data[new_column].isnull().sum() > 0:
            if self.imputation_strategy == "mean":
                data[new_column].fillna(self._global_mean, inplace=True)
            elif self.imputation_strategy == "min":
                data[new_column].fillna(data[new_column].min(),
                                        inplace=True)
            elif self.imputation_strategy == "max":
                data[new_column].fillna(data[new_column].max(),
                                        inplace=True)

        return data

    def fit_transform(self, data: pd.DataFrame,
                      column_names: list,
                      target_column: str) -> pd.DataFrame:
        """Fit the encoder and transform the data

        Parameters
        ----------
        data : pd.DataFrame
            Data to be encoded
        column_names : list
            Columns of data to be encoded
        target_column : str
            Column name of the target

        Returns
        -------
        pd.DataFrame
            data with additional discretized variables
        """
        self.fit(data, column_names, target_column)
        return self.transform(data, column_names)

    @staticmethod
    def _clean_column_name(column_name: str) -> str:
        """Clean column name string by removing "_bin" and adding "_enc"

        Parameters
        ----------
        column_name : str
            column name to be cleaned

        Returns
        -------
        str
            cleaned column name
        """
        if "_bin" in column_name:
            return column_name.replace("_bin", "") + "_enc"
        elif "_processed" in column_name:
            return column_name.replace("_processed", "") + "_enc"
        elif "_cleaned" in column_name:
            return column_name.replace("_cleaned", "") + "_enc"
        else:
            return column_name + "_enc"
