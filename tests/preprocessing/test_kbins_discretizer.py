from contextlib import contextmanager
import pytest

import numpy as np
import pandas as pd
import math

from cobra.preprocessing.kbins_discretizer import KBinsDiscretizer


@contextmanager
def does_not_raise():
    yield


class TestKBinsDiscretizer:

    # ---------------- Test for public methods ----------------
    def test_attributes_to_dict(self):

        discretizer = KBinsDiscretizer()

        bins = [(0.0, 3.0), (3.0, 6.0), (6.0, 9.0)]
        discretizer._bins_by_column = {"variable": bins}

        actual = discretizer.attributes_to_dict()

        expected = {
            "n_bins": 10,
            "strategy": "quantile",
            "closed": "right",
            "auto_adapt_bins": False,
            "starting_precision": 0,
            "label_format": "{} - {}",
            "change_endpoint_format": False,
            "_bins_by_column": {"variable": [[0.0, 3.0], [3.0, 6.0],
                                             [6.0, 9.0]]}
        }

        assert actual == expected

    @pytest.mark.parametrize("attribute",
                             ["n_bins", "strategy", "closed",
                              "auto_adapt_bins", "starting_precision",
                              "label_format", "change_endpoint_format",
                              "_bins_by_column"])
    def test_set_attributes_from_dict(self, attribute):

        discretizer = KBinsDiscretizer()

        params = {
            "n_bins": 5,
            "strategy": "uniform",
            "closed": "left",
            "auto_adapt_bins": True,
            "starting_precision": 1,
            "label_format": "[,)",
            "change_endpoint_format": True,
            "_bins_by_column": {"variable": [[0.0, 3.0], [3.0, 6.0],
                                             [6.0, 9.0]]}
        }

        expected = params[attribute]

        if attribute == "_bins_by_column":
            # list of list is transformed to a list of tuples
            # in KBinsDiscretizer!!!
            expected = {"variable": [(0.0, 3.0), (3.0, 6.0), (6.0, 9.0)]}

        discretizer.set_attributes_from_dict(params)

        actual = getattr(discretizer, attribute)

        assert actual == expected

    # no further tests here as this is just a wrapper around _fit_column!
    @pytest.mark.parametrize("strategy, expectation",
                             [("trees", pytest.raises(ValueError)),
                              ("quantile", does_not_raise())])
    def test_fit_exception(self, strategy, expectation):
        discretizer = KBinsDiscretizer(strategy=strategy)

        data = pd.DataFrame({"variable": list(range(0, 10)) + [np.nan]})

        with expectation:
            discretizer.fit(data, ["variable"])

    # no further tests here as this is just a wrapper around _transform_column!
    @pytest.mark.parametrize("scenario, expectation",
                             [("raise", pytest.raises(ValueError)),
                              ("regular_test", does_not_raise()),
                              ("constant_data", does_not_raise())])
    def test_transform(self, scenario, expectation):

        discretizer = KBinsDiscretizer(n_bins=3, strategy="uniform")

        data = pd.DataFrame({"variable": ([1] * 10)})
        expected = data.copy()

        if scenario == "regular_test":
            # overwrite data and expected with DataFrame containing
            # a non-constant variable
            data = pd.DataFrame({"variable": list(range(0, 10)) + [np.nan]})
            expected = data.copy()

            discretizer.fit(data, ["variable"])

            categories = ["0.0 - 3.0", "3.0 - 6.0", "6.0 - 9.0", "Missing"]
            expected["variable_bin"] = pd.Categorical(["0.0 - 3.0"]*4
                                                      + ["3.0 - 6.0"]*3
                                                      + ["6.0 - 9.0"]*3
                                                      + ["Missing"],
                                                      categories=categories,
                                                      ordered=True)
        elif scenario == "constant_data":
            discretizer.fit(data, ["variable"])

        with expectation:
            actual = discretizer.transform(data, ["variable"])
            pd.testing.assert_frame_equal(actual, expected)

    # ---------------- Test for private methods ----------------
    @pytest.mark.parametrize("n_bins, expectation",
                             [(1, pytest.raises(ValueError)),
                              (10.5, pytest.raises(ValueError)),
                              (2, does_not_raise())])
    def test_validate_n_bins_exception(self, n_bins, expectation):
        with expectation:
            assert KBinsDiscretizer()._validate_n_bins(n_bins=n_bins) is None

    def test_transform_column(self):

        data = pd.DataFrame({"variable": list(range(0, 10)) + [np.nan]})
        discretizer = KBinsDiscretizer(n_bins=3, strategy="uniform")

        bins = [(0.0, 3.0), (3.0, 6.0), (6.0, 9.0)]

        actual = discretizer._transform_column(data, "variable", bins)

        categories = ["0.0 - 3.0", "3.0 - 6.0", "6.0 - 9.0", "Missing"]

        expected = pd.DataFrame({"variable": list(range(0, 10)) + [np.nan]})
        expected["variable_bin"] = pd.Categorical(["0.0 - 3.0"]*4
                                                  + ["3.0 - 6.0"]*3
                                                  + ["6.0 - 9.0"]*3
                                                  + ["Missing"],
                                                  categories=categories,
                                                  ordered=True)

        # assert using pandas testing module
        pd.testing.assert_frame_equal(actual, expected)

    @pytest.mark.parametrize("n_bins, auto_adapt_bins, data, expected",
                             [(4, False,
                               pd.DataFrame({"variable": list(range(0, 11))}),
                               [(0.0, 2.0), (2.0, 5.0), (5.0, 8.0),
                                (8.0, 10.0)]),
                              (10, True,
                               # ints from 0-10 with 17 nan's
                               pd.DataFrame({"variable": list(range(0, 11)) +
                                            ([np.nan] * 17)}),
                               [(0.0, 2.0), (2.0, 5.0), (5.0, 8.0),
                                (8.0, 10.0)]),
                              (10, False,
                               # almost constant
                               pd.DataFrame({"variable": [0] + ([1] * 100)}),
                               None)],
                             ids=["regular", "auto_adapt_bins",
                                  "two bin edges"])
    def test_fit_column(self, n_bins, auto_adapt_bins, data, expected):
        discretizer = KBinsDiscretizer(n_bins=n_bins,
                                       auto_adapt_bins=auto_adapt_bins)

        actual = discretizer._fit_column(data, column_name="variable")

        assert actual == expected

    @pytest.mark.parametrize("strategy, n_bins, data, expected",
                             [("quantile",  # strategy
                               4,  # n_bins
                               # data (ints from 0 - 10):
                               pd.DataFrame({"variable": list(range(0, 11))}),
                               [0.0, 2.5, 5, 7.5, 10.0]),  # expected result
                              ("uniform",  # strategy
                               3,  # n_bins
                               # data (ints from 0 - 9):
                               pd.DataFrame({"variable": list(range(0, 10))}),
                               [0.0, 3.0, 6.0, 9.0])],  # expected result
                             ids=["quantile", "uniform"])
    def test_compute_bin_edges(self, strategy, n_bins, data, expected):

        discretizer = KBinsDiscretizer(strategy=strategy)

        actual = discretizer._compute_bin_edges(data, column_name="variable",
                                                n_bins=n_bins,
                                                col_min=data.variable.min(),
                                                col_max=data.variable.max())

        assert actual == expected

    @pytest.mark.parametrize("bin_edges, starting_precision, expected",
                             [([-10, 0, 1, 2], 1, 1),
                              ([-10, 0, 1, 1.01], 0, 2),
                              ([-10, 0, 1, 1.1], 1, 1),
                              ([-10, 0, 1, 2], -1, 0),
                              ([-10, 0, 10, 21], -1, -1)],
                             ids=["less precision", "more precision",
                                  "equal precision", "negative start",
                                  "round up"])
    def test_compute_minimal_precision_of_bin_edges(self, bin_edges,
                                                    starting_precision,
                                                    expected):

        discretizer = KBinsDiscretizer(starting_precision=starting_precision)

        actual = discretizer._compute_minimal_precision_of_bin_edges(bin_edges)

        assert actual == expected

    @pytest.mark.parametrize("bin_edges, expected",
                             [([0, 1, 1.5, 2], [(0, 1), (1, 1.5), (1.5, 2)]),
                              ([0, 1, 1.5, 3], [(0, 1), (1, 2), (2, 3)]),
                              ([np.inf, 0.0, -np.inf],
                               [(np.inf, 0.0), (0.0, -np.inf)])])
    def test_compute_bins_from_edges(self, bin_edges, expected):

        discretizer = KBinsDiscretizer()
        actual = discretizer._compute_bins_from_edges(bin_edges)

        assert actual == expected

    @pytest.mark.parametrize("change_endpoint_format, closed, bins, expected",
                             [(False, "right", [(0, 1), (1, 2), (2, 3)],
                               ["0 - 1", "1 - 2", "2 - 3"]),
                              (True, "right", [(0, 1), (1, 2), (2, 3)],
                               ["<= 1", "1 - 2", "> 2"]),
                              (True, "left", [(0, 1), (1, 2), (2, 3)],
                               ["< 1", "1 - 2", ">= 2"])],
                             ids=["standard format", "different endpoints",
                                  "different endpoints left"])
    def test_create_bin_labels(self, change_endpoint_format, closed,
                               bins, expected):

        discretizer = KBinsDiscretizer(
            closed=closed,
            change_endpoint_format=change_endpoint_format
        )

        actual = discretizer._create_bin_labels(bins)

        assert actual == expected
