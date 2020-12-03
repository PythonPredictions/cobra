"""
This module is a rework of
https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/preprocessing/_discretization.py
However, it is purely written in pandas instead of numpy because it is more
intuitive

Also, some custom modifications were included to allign it with our
Python Predictions methodology

Authors:

- Geert Verstraeten (methodology)
- Matthias Roels (implementation)
"""
# standard lib imports
from copy import deepcopy
from typing import List
import numbers

import logging
log = logging.getLogger(__name__)

# third party imports
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
#from sklearn.cluster import KMeans


class KBinsDiscretizer(BaseEstimator):

    """Bin continuous data into intervals of predefined size. It provides a
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
        ``<= x`` and ``> y`` resp.
    closed : str
        Whether to close the bins (intervals) from the left or right
    label_format : str
        format string to display the bin labels
        e.g. ``min - max``, ``(min, max]``, ...
    n_bins : int
        Number of bins to produce. Raises ValueError if ``n_bins < 2``.
    starting_precision : int
        Initial precision for the bin edges to start from,
        can also be negative. Given a list of bin edges, the class will
        automatically choose the minimal precision required to have proper bins
        e.g. ``[5.5555, 5.5744, ...]`` will be rounded to
        ``[5.56, 5.57, ...]``. In case of a negative number, an attempt will be
        made to round up the numbers of the bin edges e.g. ``5.55 -> 10``,
        ``146 -> 100``, ...
    strategy : str
        Binning strategy. Currently only `uniform` and `quantile`
        e.g. equifrequency is supported
    """

    valid_strategies = ("uniform", "quantile")
    valid_keys = ["n_bins", "strategy", "closed", "auto_adapt_bins",
                  "starting_precision", "label_format",
                  "change_endpoint_format"]

    def __init__(self, n_bins: int=10, strategy: str="quantile",
                 closed: str="right",
                 auto_adapt_bins: bool=False,
                 starting_precision: int=0,
                 label_format: str="{} - {}",
                 change_endpoint_format: bool=False):

        # validate number of bins
        self._validate_n_bins(n_bins)

        self.n_bins = n_bins
        self.strategy = strategy.lower()
        self.closed = closed.lower()
        self.auto_adapt_bins = auto_adapt_bins
        self.starting_precision = starting_precision
        self.label_format = label_format
        self.change_endpoint_format = change_endpoint_format

        # dict to store fitted output in
        self._bins_by_column = {}

    def _validate_n_bins(self, n_bins: int):
        """Check if ``n_bins`` is of the proper type and if it is bigger
        than two

        Parameters
        ----------
        n_bins : int
            Number of bins KBinsDiscretizer has to produce for each variable

        Raises
        ------
        ValueError
            in case ``n_bins`` is not an integer or if ``n_bins < 2``
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

        params["_bins_by_column"] = {
            key: [list(tup) for tup in value] if value else None
            for key, value in self._bins_by_column.items()
        }

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
            In case `_bins_by_column` is not of type dict
        """
        _bins_by_column = params.pop("_bins_by_column", {})

        if type(_bins_by_column) != dict:
            raise ValueError("_bins_by_column is expected to be a dict "
                             "but is of type {} instead"
                             .format(type(_bins_by_column)))

        # Clean out params dictionary to remove unknown keys (for safety!)
        params = {key: params[key] for key in params if key in self.valid_keys}

        # We cannot turn this method into a classmethod as we want to make use
        # of the following method from BaseEstimator:
        self.set_params(**params)

        self._bins_by_column = {
            key: ([tuple(l) for l in value] if value else None)
            for key, value in _bins_by_column.items()
        }

        return self

    def fit(self, data: pd.DataFrame, column_names: list):
        """Fits the estimator

        Parameters
        ----------
        data : pd.DataFrame
            Data to be discretized
        column_names : list
            Names of the columns of the DataFrame to discretize
        """

        if self.strategy not in self.valid_strategies:
            raise ValueError("{}: valid options for 'strategy' are {}. "
                             "Got strategy={!r} instead."
                             .format(KBinsDiscretizer.__name__,
                                     self.valid_strategies, self.strategy))

        for column_name in column_names:

            if column_name not in data.columns:
                log.warning("DataFrame has no column '{}', so it will be "
                            "skipped in fitting" .format(column_name))
                continue

            bins = self._fit_column(data, column_name)

            # Add to bins_by_column for later use
            self._bins_by_column[column_name] = bins

    def _fit_column(self, data: pd.DataFrame,
                    column_name: str) -> List[tuple]:
        """Compute bins for a specific column in data

        Parameters
        ----------
        data : pd.DataFrame
            Data to be discretized
        column_name : str
            Name of the column of the DataFrame to discretize

        Returns
        -------
        List[tuple]
            list of bins as tuples
        """

        col_min, col_max = data[column_name].min(), data[column_name].max()

        if col_min == col_max:
            log.warning("Predictor '{}' is constant and "
                        "will be ignored in computation".format(column_name))
            return None

        n_bins = self.n_bins
        if self.auto_adapt_bins:
            size = len(data.index)
            missing_pct = data[column_name].isnull().sum()/size
            n_bins = int(max(round((1 - missing_pct) * n_bins), 2))

        bin_edges = self._compute_bin_edges(data, column_name, n_bins,
                                            col_min, col_max)

        if len(bin_edges) < 3:
            log.warning("Only 1 bin was found for predictor '{}' so it will "
                        "be ignored in computation".format(column_name))
            return None

        if len(bin_edges) < n_bins + 1:
            log.warning("The number of actual bins for predictor '{}' is {} "
                        "which is smaller than the requested number of bins "
                        "{}".format(column_name, len(bin_edges) - 1, n_bins))

        return self._compute_bins_from_edges(bin_edges)

    def transform(self, data: pd.DataFrame,
                  column_names: list) -> pd.DataFrame:
        """Discretizes the data in the given list of columns by mapping each
        number to the appropriate bin computed by the fit method

        Parameters
        ----------
        data : pd.DataFrame
            Data to be discretized
        column_names : list
            Names of the columns of the DataFrame to discretize

        Returns
        -------
        pd.DataFrame
            data with additional discretized variables
        """
        if len(self._bins_by_column) == 0:
            msg = ("{} instance is not fitted yet. Call 'fit' with "
                   "appropriate arguments before using this method.")

            raise NotFittedError(msg.format(self.__class__.__name__))

        for column_name in column_names:
            if column_name not in self._bins_by_column:
                log.warning("Column '{}' is not in fitted output "
                            "and will be skipped".format(column_name))
                continue

            # can be None for a column with a constant value!
            bins = self._bins_by_column[column_name]
            if bins is not None:
                data = self._transform_column(data, column_name, bins)

        return data

    def _transform_column(self, data: pd.DataFrame,
                          column_name: str,
                          bins: List[tuple]) -> pd.DataFrame:
        """Given a DataFrame, a column name and a list of bins,
        create an additional column which determines the bin in which the value
        of column_name lies in.

        Parameters
        ----------
        data : pd.DataFrame
            Original data to be discretized
        column_name : str
            name of the column to discretize
        bins : List[tuple]
            bins to discretize the data into

        Returns
        -------
        pd.DataFrame
            original DataFrame with an added binned column
        """

        interval_idx = KBinsDiscretizer._create_index(bins, self.closed)

        column_name_bin = column_name + "_bin"

        # use pd.cut to compute bins
        data.loc[:, column_name_bin] = pd.cut(x=data[column_name],
                                              bins=interval_idx)

        # Rename bins so that the output has a proper format
        bin_labels = self._create_bin_labels(bins)

        data.loc[:, column_name_bin] = (data[column_name_bin]
                                        .cat.rename_categories(bin_labels))

        if data[column_name_bin].isnull().sum() > 0:

            # Add an additional bin for missing values
            data[column_name_bin].cat.add_categories(["Missing"], inplace=True)

            # Replace NULL with "Missing"
            # Otherwise these will be ignored in groupby
            data[column_name_bin].fillna("Missing", inplace=True)

        return data

    def fit_transform(self, data: pd.DataFrame,
                      column_names: list) -> pd.DataFrame:
        """Fits to data, then transform it

        Parameters
        ----------
        data : pd.DataFrame
            Data to be discretized
        column_names : list
            Names of the columns of the DataFrame to discretize

        Returns
        -------
        pd.DataFrame
            data with additional discretized variables
        """
        self.fit(data, column_names)
        return self.transform(data, column_names)

    def _compute_bin_edges(self, data: pd.DataFrame, column_name: str,
                           n_bins: int, col_min: float,
                           col_max: float) -> list:
        """Compute the bin edges for a given column, a DataFrame and the number
        of required bins

        Parameters
        ----------
        data : pd.DataFrame
            Data to be discretized
        column_name : str
             name of the column to discretize
        n_bins : int
            Number of bins to produce.
        col_min : float
            min value of the variable
        col_max : float
            max value of the variable

        Returns
        -------
        list
            list of bin edges from which to compute the bins
        """

        bin_edges = []
        if self.strategy == "quantile":
            bin_edges = list(data[column_name]
                             .quantile(np.linspace(0, 1, n_bins + 1),
                                       interpolation='linear'))
        elif self.strategy == "uniform":
            bin_edges = list(np.linspace(col_min, col_max, n_bins + 1))

        # elif self.strategy == "kmeans":

        #     if data[column_name].isnull().sum() > 0:
        #         raise ValueError("{}: kmeans strategy cannot handle NULL "
        #                          "values in the data."
        #                          .format(KBinsDiscretizer.__name__))

        #     # Deterministic initialization with uniform spacing
        #     uniform_edges = np.linspace(col_min, col_max, n_bins + 1)
        #     init = (uniform_edges[1:] + uniform_edges[:-1])[:, None] * 0.5

        #     # 1D k-means
        #     kmeans = KMeans(n_clusters=n_bins, init=init, n_init=1)
        #     centers = (kmeans.fit(data[column_name][:, None])
        #                      .cluster_centers_[:, 0])

        #     # Make sure to sort centers as they may be unsorted,
        #     # even with sorted init!
        #     centers.sort()

        #     # compute bin_edges from centers
        #     bin_edges = (centers[1:] + centers[:-1]) * 0.5
        #     bin_edges = np.r_[col_min, bin_edges, col_max]

        # Make sure the bin_edges are unique and sorted
        return sorted(list(set(bin_edges)))

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

    def _compute_bins_from_edges(self, bin_edges: list) -> List[tuple]:
        """Given a list of bin edges, compute the minimal precision for which
        we can make meaningful bins and make those bins

        Parameters
        ----------
        bin_edges : list
            The bin edges for binning a continuous variable

        Returns
        -------
        List[tuple]
            A (sorted) list of bins as tuples
        """
        # compute the minimal precision of the bin_edges
        # this can be a negative number, which then
        # rounds numbers to the nearest 10, 100, ...
        precision = self._compute_minimal_precision_of_bin_edges(bin_edges)

        bins = []
        for a, b in zip(bin_edges, bin_edges[1:]):
            fmt_a = round(a, precision)
            fmt_b = round(b, precision)

            bins.append((fmt_a, fmt_b))

        return bins

    @staticmethod
    def _create_index(intervals: List[tuple],
                      closed: str="right") -> pd.IntervalIndex:
        """Create an pd.IntervalIndex based on a list of tuples.
        This is basically a wrapper around pd.IntervalIndex.from_tuples
        However, the lower bound of the first entry in the list (the lower bin)
        is replaced by -np.inf. Similarly, the upper bound of the last entry in
        the list (upper bin) is replaced by np.inf.

        Parameters
        ----------
        intervals : List[tuple]
            a list of tuples describing the intervals
        closed : str, optional
            Whether the intervals should be closed on the left-side,
            right-side, both or neither.

        Returns
        -------
        pd.IntervalIndex
            Description
        """

        # check if closed is of the proper form
        if closed not in ["left", "right"]:
            raise ValueError("{}: valid options for 'closed' are {}. "
                             "Got strategy={!r} instead."
                             .format(KBinsDiscretizer.__name__,
                                     ["left", "right"], closed))

        # deepcopy variable because we do not want to modify the content
        # of intervals (which is still used outside of this function)
        _intervals = deepcopy(intervals)
        # Replace min and max with -np.inf and np.inf resp. so that these
        # values are guaranteed to be included when transforming the data
        _intervals[0] = (-np.inf, _intervals[0][1])
        _intervals[-1] = (_intervals[-1][0], np.inf)

        return pd.IntervalIndex.from_tuples(_intervals, closed)

    def _create_bin_labels(self, bins: List[tuple]) -> list:
        """Given a list of bins, create a list of string containing the bins
        as a string with a specific format (e.g. bin labels)

        Parameters
        ----------
        bins : List[tuple]
            list of tuple containing for each bin the upper and lower bound

        Returns
        -------
        list
            list of (formatted) bin labels
        """
        bin_labels = []
        for interval in bins:
            bin_labels.append(self.label_format.format(interval[0],
                                                       interval[1]))

        # Format first and last bin as < x and > y resp.
        if self.change_endpoint_format:
            if self.closed == "left":
                bin_labels[0] = "< {}".format(bins[0][1])
                bin_labels[-1] = ">= {}".format(bins[-1][0])
            else:
                bin_labels[0] = "<= {}".format(bins[0][1])
                bin_labels[-1] = "> {}".format(bins[-1][0])

        return bin_labels
