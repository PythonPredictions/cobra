"""
This module is a rework of
https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/preprocessing/
_discretization.py
However, it is written in PySpark instead of numpy for big data processing.
Also, some custom modifications were included to allign it with our methodology
Authors:
- Geert Verstraeten (methodology)
- Matthias Roels (implementation)
"""
# standard lib imports
from copy import deepcopy
from itertools import chain
import numbers

import logging
log = logging.getLogger(__name__)

# third party imports
import numpy as np

from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError

from pyspark.sql import DataFrame
from pyspark.ml.feature import Bucketizer
from pyspark.sql.functions import lit
from pyspark.sql.functions import col
from pyspark.sql.functions import when
from pyspark.sql.functions import sum as spark_sum
from pyspark.sql.functions import create_map


class KBinsDiscretizer(BaseEstimator):

    """Bin continuous data into intervals of predefined size. This provides a
    way to partition continuous data into discrete values, i.e. tranform
    continuous data into nominal data. This can make a linear model more
    expressive as it introduces nonlinearity to the model, while maintaining
    the interpretability of the model afterwards.
    Attributes
    ----------
    auto_adapt_bins : bool
        reduces the number of bins (starting from n_bins) as a function of
        the number of missings
    change_endpoint_format : bool
        Whether or not to change the format of the lower and upper bins into
        "< x" and "> y" resp.
    label_format : str
        format string to display the bin labels e.g. min - max, (min, max], ...
    n_bins : int
        Number of bins to produce. Raises ValueError if ``n_bins < 2``.
    relative_error: float
        the relative computation error in approxQuantile to compute the
        percentile of a predictor
    starting_precision : int
        Initial precision for the bin edges to start from,
        can also be negative. Given a list of bin edges, the class will
        automatically choose the minimal precision required to have proper bins
        e.g. [5.5555, 5.5744, ...] will be rounded to [5.56, 5.57, ...]. In
        case of a negative number, an attempt will be made to round up the
        numbers of the bin edges e.g. 5.55 -> 10, 146 -> 100, ...
    """

    valid_keys = ["n_bins", "auto_adapt_bins", "relative_error",
                  "starting_precision", "label_format",
                  "change_endpoint_format"]

    def __init__(self, n_bins: int=10, relative_error: float=0.01,
                 auto_adapt_bins: bool=False,
                 starting_precision: int=0,
                 label_format: str="{} - {}",
                 change_endpoint_format: bool=False):

        # validate number of bins
        self._validate_n_bins(n_bins)

        self.n_bins = n_bins
        self.relative_error = relative_error
        self.auto_adapt_bins = auto_adapt_bins
        self.starting_precision = starting_precision
        self.label_format = label_format
        self.change_endpoint_format = change_endpoint_format

        # dict to store fitted output in
        self._bin_edges_by_column = {}
        self._bin_labels_by_column = {}

    def _validate_n_bins(self, n_bins: int):
        """Check if n_bins is of the proper type and if it is bigger than two
        Parameters
        ----------
        n_bins : int
            Number of bins KBinsDiscretizer has to produce for each variable
        Raises
        ------
        ValueError
            in case n_bins is not an integer or if n_bins < 2
        """
        if not isinstance(n_bins, numbers.Integral):
            raise ValueError("{} received an invalid n_bins type. "
                             "Received {}, expected int."
                             .format(KBinsDiscretizer.__name__,
                                     type(n_bins).__name__))
        if n_bins < 2:
            raise ValueError("{} received an invalid number "
                             "of bins. Received {}, expected at least 2."
                             .format(KBinsDiscretizer.__name__, n_bins))

    def attributes_to_dict(self) -> dict:
        """Return the attributes of KBinsDiscretizer in a dictionary
        Returns
        -------
        dict
            Contains the attributes of KBinsDiscretizer instance with the names
            as keys
        """
        params = self.get_params()

        params["_bin_edges_by_column"] = self._bin_edges_by_column
        params["_bin_labels_by_column"] = self._bin_labels_by_column

        return params

    def set_attributes_from_dict(self, params: dict):
        """Set instance attributes from a dictionary of values with key the
        name of the attribute.
        Parameters
        ----------
        params : dict
            Contains the attributes of KBinsDiscretizer with their
            names as key.
        Raises
        ------
        ValueError
            In case _bins_by_column is not of type dict
        """
        self._bin_edges_by_column = params.pop("_bin_edges_by_column", {})
        self._bin_labels_by_column = params.pop("_bin_labels_by_column", {})

        if type(self._bin_edges_by_column) != dict:
            raise ValueError("_bin_edges_by_column is expected to be a dict "
                             "but is of type {} instead"
                             .format(type(self._bin_edges_by_column)))
        elif type(self._bin_labels_by_column) != dict:
            raise ValueError("_bin_labels_by_column is expected to be a dict "
                             "but is of type {} instead"
                             .format(type(self._bin_labels_by_column)))

        # Clean out params dictionary to remove unknown keys (for safety!)
        params = {key: params[key] for key in params if key in self.valid_keys}

        # We cannot turn this method into a classmethod as we want to make use
        # of the following method from BaseEstimator:
        self.set_params(**params)

        return self

    def fit(self, data: DataFrame, column_names: list):
        """Fits the estimator
        Parameters
        ----------
        data : DataFrame
            Data to be discretized
        column_names : list
            Columns of data to be discretized
        """

        n_bins_by_column = self._compute_number_of_bins_by_column(
            data,
            column_names
        )

        for column_name in column_names:

            if column_name not in data.columns:
                log.warning("DataFrame has no column '{}', so it will be "
                            "skipped in fitting" .format(column_name))
                continue

            n_bins = n_bins_by_column[column_name]
            bin_edges = self._compute_bin_edges(data, column_name, n_bins,
                                                self.relative_error)

            if len(bin_edges) < 3:
                log.warning("Only 1 bin was found for predictor '{}' so it "
                            "will be ignored in computation"
                            .format(column_name))
                continue

            if len(bin_edges) < n_bins + 1:
                log.warning("The number of actual bins for predictor '{}' is "
                            "{} which is smaller than the requested number of "
                            "bins {}".format(column_name, len(bin_edges) - 1,
                                             n_bins))

            bin_labels = self._create_bin_labels_from_edges(bin_edges)

            self._bin_edges_by_column[column_name] = bin_edges
            self._bin_labels_by_column[column_name] = bin_labels

    def transform(self, data: DataFrame,
                  column_names: list) -> DataFrame:
        """Discretizes the data in the given list of columns by mapping each
        number to the appropriate bin computed by the fit method
        Parameters
        ----------
        data : DataFrame
            Data to be discretized
        column_names : list
            Columns of data to be discretized
        Returns
        -------
        DataFrame
            data with additional discretized variables
        """
        if len(self._bin_edges_by_column) == 0:
            msg = ("{} instance is not fitted yet. Call 'fit' with "
                   "appropriate arguments before using this method.")

            raise NotFittedError(msg.format(self.__class__.__name__))

        for column_name in column_names:
            if column_name not in self._bin_edges_by_column:
                log.warning("Column '{}' is not in fitted output "
                            "and will be skipped".format(column_name))
                continue

            bin_edges = self._bin_edges_by_column[column_name]
            bin_labels = self._bin_labels_by_column[column_name]

            data = (KBinsDiscretizer.
                    _transform_column(data, column_name, bin_edges,
                                      bin_labels))

        return data.select([c for c in data.columns
                            if not c.endswith("__bucket")])

    @staticmethod
    def _transform_column(data: DataFrame,
                          column_name: str,
                          bin_edges: list,
                          bin_labels: list) -> DataFrame:
        """Given a DataFrame, a column name and a list of bins,
        create an additional column which determines the bin in which the value
        of column_name lies in.

        Parameters
        ----------
        data : DataFrame
            Original data to be discretized
        column_name : str
            name of the column to discretize
        bin_edges : list
            bin edges to be used for discretization
        bin_labels : list
            formatted bin labels
        Returns
        -------
        DataFrame
            original DataFrame with an added binned column
        """

        _bin_edges = deepcopy(bin_edges)
        _bin_edges[0] = -np.inf
        _bin_edges[-1] = np.inf

        bucketizer = Bucketizer(splits=_bin_edges,
                                inputCol=column_name,
                                outputCol=column_name + "__bucket",
                                handleInvalid="keep")

        data = bucketizer.transform(data)

        mapping = {float(i): bin_labels[i] for i in range(len(bin_labels))}
        mapping_expr = create_map([lit(x) for x in chain(*mapping.items())])

        data = (data
                .withColumn(column_name + "_bin",
                            when(col(column_name + "__bucket").isNull(),
                                 "Missing")
                            .otherwise(mapping_expr
                                       .getItem(col(column_name + "__bucket")))
                            ))

        return data

    def fit_transform(self, data: DataFrame,
                      column_names: list) -> DataFrame:
        """Summary
        Parameters
        ----------
        data : DataFrame
            Data to be discretized
        column_names : list
            Columns of data to be discretized
        Returns
        -------
        DataFrame
            data with additional discretized variables
        """
        self.fit(data, column_names)
        return self.transform(data, column_names)

    def _compute_number_of_bins_by_column(self, data: DataFrame,
                                          column_names: list) -> dict:
        """Summary

        Parameters
        ----------
        data : DataFrame
            Data to be discretized
        column_names : list
            Columns of data to be discretized
        """

        # default is no modification to the number of bins
        n_bins_by_column = {c: self.n_bins for c in column_names}
        if self.auto_adapt_bins:
            prop_missings_by_column = (KBinsDiscretizer
                                       ._compute_prop_missing_per_column(
                                           data,
                                           column_names))

            n_bins_by_column = {
                c: int(max(round((1-prop_missings_by_column[c])*self.n_bins),
                           2))
                for c in column_names
            }

        return n_bins_by_column

    def _compute_bin_edges(self, data: DataFrame, column_name: str,
                           n_bins: int, relative_error: float) -> list:
        """Compute the bin edges for a given column, a DataFrame and the number
        of required bins

        Parameters
        ----------
        data : DataFrame
            Data to be discretized
        column_name : str
             name of the column to discretize
        n_bins : int
            Number of bins to produce.
        relative_error : float
            Description
        Returns
        -------
        list
            list of bin edges from which to compute the bins
        """
        bin_edges = data.approxQuantile(
            col=column_name,
            probabilities=list(np.linspace(0, 1, n_bins + 1)),
            relativeError=relative_error)

        bin_edges = sorted(list(set(bin_edges)))

        precision = self._compute_minimal_precision_of_bin_edges(bin_edges)

        return [round(edge, precision) for edge in bin_edges]

    def _compute_minimal_precision_of_bin_edges(self, bin_edges: list) -> int:
        """Compute the minimal precision of a list of bin_edges so that we end
        up with a strictly ascending sequence of numbers.
        The starting_precision attribute will be used as the initial precision.
        In case of a negative starting_precision, the bin edges will be rounded
        to the nearest 10, 100, ... (e.g. 5.55 -> 10, 246 -> 200, ...)
        Parameters
        ----------
        bin_edges : list
            The bin edges for binning a continuous variable
        Returns
        -------
        int
            minimal precision for the bin edges
        """

        precision = self.starting_precision
        while True:
            cont = False
            for a, b in zip(bin_edges, bin_edges[1:]):
                if a != b and round(a, precision) == round(b, precision):
                    # precision is not high enough, so increase
                    precision += 1
                    cont = True  # set cont to True to keep looping
                    break  # break out of the for loop
            if not cont:
                # if minimal precision was found,
                # return to break out of while loop
                return precision

    def _create_bin_labels_from_edges(self, bin_edges: list) -> list:
        """Given a list of bin edges, create a list of strings containing the
        bin labels with a specific format
        Parameters
        ----------
        bins : list
            list containing the bin edges
        Returns
        -------
        list
            list of (formatted) bin labels
        """
        bin_labels = []
        for interval in zip(bin_edges, bin_edges[1:]):
            bin_labels.append(self.label_format.format(interval[0],
                                                       interval[1]))

        # Format first and last bin as < x and > y resp.
        if self.change_endpoint_format:
            bin_labels[0] = "<= {}".format(bin_edges[1])
            bin_labels[-1] = "> {}".format(bin_edges[-2])

        return bin_labels

    @staticmethod
    def _compute_prop_missing_per_column(data: DataFrame,
                                         column_names: list) -> dict:
        """Compute proportion of missing values per column

        Parameters
        ----------
        data : pd.DataFrame
            basetable
        column_names : list
            Column names of basetable to compute proportion of missing values
            of
        Returns
        -------
        dict
            a map of column name to proportion of missings
        """
        size = data.count()

        res = data.select(*(spark_sum(col(c).isNull().cast("int")).alias(c)
                            for c in column_names)).collect()[0]

        return {c: res[c]/size for c in column_names}
