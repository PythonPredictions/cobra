"""
Incidence Replacement Module. The implementation is inspired by
https://contrib.scikit-learn.org/categorical-encoding/index.html,
but is written in PySpark for big data purposes.
Authors:
- Geert Verstraeten (methodology)
- Matthias Roels (implementation)
"""
from itertools import chain
import logging
log = logging.getLogger(__name__)

from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError

from pyspark.sql import DataFrame
from pyspark.sql.functions import lit
from pyspark.sql.functions import col
from pyspark.sql.functions import when
from pyspark.sql.functions import create_map


class TargetEncoder(BaseEstimator):

    """Target encoding for categorical features.
    Replace each value of the categorical feature with the average of the
    target values (in case of a binary target, this is the incidence of the
    group). This encoding scheme is also called Mean encoding.
    The main problem with Target encoding is overfitting; the fact that we are
    encoding the feature based on target classes may lead to data leakage,
    rendering the feature biased. This could be solved using some type of
    regularization. A popular way to handle this is to use cross-validation
    and compute the means in each out-of-fold. This is out-of-scope for this
    module
    Attributes
    ----------
    imputation_strategy : str
        in case there is a particular column which contains new categories,
        the encoding will lead to NULL values which should be imputed.
        Valid strategies are to replace with the global mean of the train
        set or the min (resp. max) incidence of the categories of that
        particular variable.
    """

    valid_strategies = ("mean", "min", "max")

    def __init__(self, imputation_strategy: str="mean"):

        if imputation_strategy not in self.valid_strategies:
            raise ValueError("Valid options for 'imputation_strategy' are {}."
                             " Got imputation_strategy={!r} instead"
                             .format(self.valid_strategies,
                                     imputation_strategy))

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

        params["_mapping"] = self._mapping

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

        if ("imputation_strategy" in params and
                params["imputation_strategy"] in self.valid_strategies):

            self.imputation_strategy = params["imputation_strategy"]

        if "_global_mean" in params and type(params["_global_mean"]) == float:
            self._global_mean = params["_global_mean"]

        if "_mapping" in params and type(params["_mapping"]) == dict:
            self._mapping = params["_mapping"]

        return self

    def fit(self, data: DataFrame, column_names: list,
            target_column: str):
        """Fit the TargetEncoder to data and y
        Parameters
        ----------
        data : DataFrame
            data used to compute the mapping to encode the categorical
            variables with.
        column_names : list
            Columns of data to be encoded
        target_column : str
            Column name of the target
        """

        # compute global mean (target incidence in case of binary target)
        self._global_mean = (data.groupBy()
                             .agg({target_column: "mean"})
                             .collect()[0]["avg({})".format(target_column)])

        for column_name in column_names:
            if column_name not in data.columns:
                log.warning("DataFrame has no column '{}', so it will be "
                            "skipped in fitting" .format(column_name))
                continue

            res = data.groupby(column_name).agg({target_column: "mean"})

            self._mapping[column_name] = {
                row[column_name]: row["avg({})".format(target_column)]
                for row in res.collect()
            }

    def transform(self, data: DataFrame,
                  column_names: list) -> DataFrame:
        """Replace (e.g. encode) categories of each column with its average
        incidence which was computed when the fit method was called
        Parameters
        ----------
        data : DataFrame
            data to encode
        column_names : list
             Columns of data to be encoded
        Returns
        -------
        DataFrame
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

    def _transform_column(self, data: DataFrame,
                          column_name: str) -> DataFrame:
        """Replace (e.g. encode) categories of each column with its average
        incidence which was computed when the fit method was called
        Parameters
        ----------
        data : pd.DataFrame
            data to encode
        column_name : str
            Name of the column in data to be encoded
        Returns
        -------
        DataFrame
            transformed data
        """
        new_column = TargetEncoder._clean_column_name(column_name)
        mapping = self._mapping[column_name]

        # In case of categorical data, it could be that new categories will
        # emerge which were not present in the train set, so this would result
        # in missing values in the encoded variables. These should be replaced
        # with appropriate values based on the imputation strategy
        replace_value = None
        if self.imputation_strategy == "mean":
            replace_value = self._global_mean
        elif self.imputation_strategy == "min":
            replace_value = mapping[min(mapping, key=mapping.get)]
        elif self.imputation_strategy == "max":
            replace_value = mapping[max(mapping, key=mapping.get)]

        # Create mapping expression to map the values the column to their
        # respective incidences.
        mapping_expr = create_map([lit(x) for x in chain(*mapping.items())])

        data = (data
                .withColumn(new_column,
                            when(~col(column_name).isin(set(mapping.keys())),
                                 replace_value)
                            .otherwise(mapping_expr.getItem(col(column_name))))
                )

        return data

    def fit_transform(self, data: DataFrame,
                      column_names: list,
                      target_column: str) -> DataFrame:
        """Summary
        Parameters
        ----------
        data : DataFrame
            Data to be encoded
        column_names : list
            Columns of data to be encoded
        target_column : str
            Column name of the target
        Returns
        -------
        DataFrame
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
