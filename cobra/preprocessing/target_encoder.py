
import logging

import pandas as pd
from tqdm.auto import tqdm
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError

log = logging.getLogger(__name__)

class TargetEncoder(BaseEstimator):
    """Target encoding for categorical features, inspired by
    http://contrib.scikit-learn.org/category_encoders/targetencoder.html.

    Replace each value of the categorical feature with the average of the
    target values (in case of a binary target, this is the incidence of the
    group). This encoding scheme is also called Mean encoding.

    Note that, when applying this target encoding, values of the categorical
    feature that have not been seen during fit will be imputed according to the
    configured imputation strategy: replacement with the mean, minimum or
    maximum value of the categorical variable.

    The main problem with Target encoding is overfitting; the fact that we are
    encoding the feature based on target classes may lead to data leakage,
    rendering the feature biased.
    This can be solved using some type of regularization. A popular way to
    handle this is to use cross-validation and compute the means in each
    out-of-fold. However, the approach implemented here makes use of
    additive smoothing (https://en.wikipedia.org/wiki/Additive_smoothing).

    In summary:

    - with a binary classification target, a value of a categorical variable is
    replaced with:

        [count(variable=value) * P(target=1|variable=value) + weight * P(target=1)]
        / [count(variable=value) + weight]

    - with a regression target, a value of a categorical variable is replaced
    with:

        [count(variable=value) * E(target|variable=value) + weight * E(target)]
        / [count(variable=value) + weight]

    Attributes
    ----------
    imputation_strategy : str
        In case there is a particular column which contains new categories,
        the encoding will lead to NULL values which should be imputed.
        Valid strategies then are to replace the NULL values with the global
        mean of the train set or the min (resp. max) incidence of the
        categories of that particular variable.
    weight : float
        Smoothing parameter (non-negative). The higher the value of the
        parameter, the bigger the contribution of the overall mean of targets
        learnt from all training data (prior) and the smaller the contribution
        of the mean target learnt from data with the current categorical value
        (posterior), so the bigger the smoothing (regularization) effect.
        When set to zero, there is no smoothing (e.g. the mean target of the
        current categorical value is used).
    """

    valid_imputation_strategies = ("mean", "min", "max")

    def __init__(self, weight: float=0.0,
                 imputation_strategy: str="mean"):

        if weight < 0:
            raise ValueError("The value of weight cannot be smaller than zero")
        elif imputation_strategy not in self.valid_imputation_strategies:
            raise ValueError("Valid options for 'imputation_strategy' are {}."
                             " Got imputation_strategy={!r} instead."
                             .format(self.valid_imputation_strategies,
                                     imputation_strategy))

        if weight == 0:
            log.warning("The target encoder's additive smoothing weight is "
                        "set to 0. This disables smoothing and may make the "
                        "encoding prone to overfitting.")

        self.weight = weight
        self.imputation_strategy = imputation_strategy

        self._mapping = {}  # placeholder for fitted output
        # placeholder for the global incidence of the data used for fitting
        self._global_mean = None

    def attributes_to_dict(self) -> dict:
        """Return the attributes of TargetEncoder in a dictionary.

        Returns
        -------
        dict
            Contains the attributes of TargetEncoder instance with the names
            as keys.
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
                params["imputation_strategy"] in self.valid_imputation_strategies):
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
        """Fit the TargetEncoder to the data.

        Parameters
        ----------
        data : pd.DataFrame
            Data used to compute the mapping to encode the categorical
            variables with.
        column_names : list
            Columns of data to be encoded.
        target_column : str
            Column name of the target.
        """
        # compute global mean (target incidence in case of binary target)
        y = data[target_column]
        self._global_mean = y.sum() / y.count()

        for column in tqdm(column_names, desc="Fitting target encoding..."):
            if column not in data.columns:
                log.warning("DataFrame has no column '{}', so it will be "
                            "skipped in fitting" .format(column))
                continue

            self._mapping[column] = self._fit_column(data[column], y)

    def _fit_column(self, X: pd.Series, y: pd.Series) -> pd.Series:
        """Replace the values of a column, holding a categorical value,
        with a new value reflecting the formulas mentioned in the docstring
        of this class.

        Parameters
        ----------
        X : pd.Series
            Data used to compute the encoding mapping for an individual
            categorical variable.
        y : pd.Series
            Series containing the targets for each observation (value) of
            this categorical variable.

        Returns
        -------
        pd.Series
            Mapping containing the new value to replace each distinct value
            of the categorical variable with.
        """
        stats = y.groupby(X).agg(["mean", "count"])

        # Note: if self.weight = 0, we have the ordinary incidence replacement
        numerator = (stats["count"] * stats["mean"]
                     + self.weight * self._global_mean)

        denominator = stats["count"] + self.weight

        return numerator / denominator

    def transform(self, data: pd.DataFrame,
                  column_names: list) -> pd.DataFrame:
        """Replace (e.g. encode) values of each categorical column with a
        new value (reflecting the corresponding average target value,
        optionally smoothed by a regularization weight),
        which was computed when the fit method was called.

        Parameters
        ----------
        data : pd.DataFrame
            Data to encode.
        column_names : list
            Name of the categorical columns in the data to be encoded.

        Returns
        -------
        pd.DataFrame
            The resulting transformed data.

        Raises
        ------
        NotFittedError
            Exception when TargetEncoder was not fitted before calling this
            method.
        """
        if (len(self._mapping) == 0) or (self._global_mean is None):
            msg = ("This {} instance is not fitted yet. Call 'fit' with "
                   "appropriate arguments before using this method.")
            raise NotFittedError(msg.format(self.__class__.__name__))

        for column in tqdm(column_names, desc="Applying target encoding..."):
            if column not in data.columns:
                log.warning("Unknown column '{}' will be skipped."
                            .format(column))
                continue
            elif column not in self._mapping:
                log.warning("Column '{}' is not in fitted output "
                            "and will be skipped.".format(column))
                continue
            data = self._transform_column(data, column)

        return data

    def _transform_column(self, data: pd.DataFrame,
                          column_name: str) -> pd.DataFrame:
        """Replace (e.g. encode) values of a categorical column with a
        new value (reflecting the corresponding average target value,
        optionally smoothed by a regularization weight),
        which was computed when the fit method was called.

        Parameters
        ----------
        data : pd.DataFrame
            Data to encode.
        column_name : str
            Name of the column in the data to be encoded.

        Returns
        -------
        pd.DataFrame
            Resulting transformed data.
        """
        new_column = TargetEncoder._clean_column_name(column_name)

        # Convert dtype to float, because when the original dtype
        # is of type "category", the resulting dtype would otherwise also be of
        # type "category":
        data[new_column] = (data[column_name].map(self._mapping[column_name])
                            .astype("float"))

        # In case of categorical data, it could be that new categories will
        # emerge which were not present in the train set, so this will result
        # in missing values, which should be replaced according to the
        # configured imputation strategy:
        if data[new_column].isnull().sum() > 0:
            if self.imputation_strategy == "mean":
                data[new_column].fillna(self._global_mean,
                                        inplace=True)
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
        """Fit the encoder and transform the data.

        Parameters
        ----------
        data : pd.DataFrame
            Data to be encoded.
        column_names : list
            Columns of data to be encoded.
        target_column : str
            Column name of the target.

        Returns
        -------
        pd.DataFrame
            Data with additional columns, holding the target-encoded variables.
        """
        self.fit(data, column_names, target_column)
        return self.transform(data, column_names)

    @staticmethod
    def _clean_column_name(column_name: str) -> str:
        """Generate a name for the new column that this target encoder
        generates in the given data, by removing "_bin", "_processed" or
        "_cleaned" from the original categorical column, and adding "_enc".

        Parameters
        ----------
        column_name : str
            Column name to be cleaned.

        Returns
        -------
        str
            Cleaned column name.
        """
        if "_bin" in column_name:
            return column_name.replace("_bin", "") + "_enc"
        elif "_processed" in column_name:
            return column_name.replace("_processed", "") + "_enc"
        elif "_cleaned" in column_name:
            return column_name.replace("_cleaned", "") + "_enc"
        else:
            return column_name + "_enc"
