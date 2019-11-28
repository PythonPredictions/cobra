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

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError


class TargetEncoder(BaseEstimator, TransformerMixin):

    """Target encoding for categorical features.

    Replace each value of the categorical feature with the average of the
    target values (in case of a binary target, this is the incidence of the
    group). This encoding scheme is also called Mean encoding.

    The main problem with Target encoding is overfitting; the fact that we are
    encoding the feature based on target classes may lead to data leakage,
    rendering the feature biased. This can be solved using some type of
    regularization. A popular way to handle this is to use cross-validation
    and compute the means in each out-of-fold. However, the approach
    implemented makes use of additive smoothing
    (https://en.wikipedia.org/wiki/Additive_smoothing)

    Attributes
    ----------
    columns : list
        A list of columns to encode, if None, all string columns will be
        encoded.
    weight : float
        Smoothing parameters (non-negative). The higher the value of the
        parameter, the bigger the contribution of the overall mean. When set to
        zero, there is no smoothing (e.g. the pure target incidence is used).
    """

    def __init__(self, columns: list=None, weight: float=0.0):

        if weight < 0:
            raise ValueError("The value of weight cannot be smaller than zero")

        self.columns = columns
        self.weight = weight
        self._mapping = {}  # placeholder for fitted output

        # not implemented yet!
        # randomized: bool=False, sigma=0.05
        # self.randomized = randomized
        # self.sigma = sigma

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Fit the TargetEncoder to X and y

        Parameters
        ----------
        X : pd.DataFrame
            data used to compute the mapping to encode the categorical
            variables with.
        y : pd.Series
            series containing the targets for each observation

        Raises
        ------
        ValueError
            if the length of X and y are not equal
        """
        # The lengths must be equal
        if len(X.index) != len(y.index):
            raise ValueError("The length of X is {}, but the length of y is {}"
                             .format(len(X.index), len(y.index)))

        if self.columns is None:
            self.columns = TargetEncoder._get_categorical_columns(X)

        # compute global mean (target incidence in case of binary target)
        global_mean = y.sum() / y.count()

        for column in self.columns:
            if column not in X.columns:
                log.warning("DataFrame has no column '{}', so it will be "
                            "skipped in fitting" .format(column))
                continue

            self._mapping[column] = self._fit_column(X[column], y, global_mean)

    def _fit_column(self, X: pd.Series, y: pd.Series,
                    global_mean: float) -> pd.Series:
        """Summary

        Parameters
        ----------
        X : pd.Series
            data used to compute the encoding mapping for an individual
            categorical variable.
        y : pd.Series
            series containing the targets for each observation
        global_mean : float
            Global mean of the target

        Returns
        -------
        pd.Series
            Mapping containing the value to replace each group of the
            categorical with.
        """
        stats = y.groupby(X).agg(["mean", "count"])

        # To do: add Gaussian noise to the estimate
        # Q: do we need to do this here or during the transform phase???

        # Note if self.weight = 0, we have the ordinary incidence replacement
        numerator = stats["count"]*stats["mean"] + self.weight*global_mean
        denominator = stats["count"] + self.weight

        return numerator/denominator

    def transform(self, X: pd.DataFrame, y: pd.Series=None) -> pd.DataFrame:
        """Replace (e.g. encode) categories of each column with its average
        incidence which was computed when the fit method was called

        Parameters
        ----------
        X : pd.DataFrame
            data to encode
        y : pd.Series, optional
            Ignored (added for compatibility with scikit-learn)

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
        if len(self._mapping) == 0:
            msg = ("This {} instance is not fitted yet. Call 'fit' with "
                   "appropriate arguments before using this method.")

            raise NotFittedError(msg.format(self.__class__.__name__))

        new_columns = []
        for column in self.columns:

            if column not in X.columns:
                log.warning("Column '{}' is not in fitted output "
                            "and will be skipped".format(column))
                continue

            new_column = TargetEncoder._clean_column_name(column)

            # Convert dtype to float because when the original dtype
            # is of type "category", the resulting dtype is also of type
            # "category"
            X[new_column] = (X[column].map(self._mapping[column])
                             .astype("float"))

            new_columns.append(new_column)

        return X[new_columns]

    @staticmethod
    def _get_categorical_columns(data: pd.DataFrame) -> list:
        """Get the columns containing categorical data
        (dtype "object" or "category")

        Parameters
        ----------
        data : pd.DataFrame
            Description

        Returns
        -------
        list
            List of column names containing categorical data
        """
        object_columns = data.dtypes[data.dtypes == object].index
        categorical_columns = data.dtypes[data.dtypes == "category"].index

        return list(set(object_columns).union(set(categorical_columns)))

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
        else:
            return column_name + "_enc"
