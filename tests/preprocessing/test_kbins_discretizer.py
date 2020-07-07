from contextlib import contextmanager
import pytest

import numpy as np
import pandas as pd

from cobra.preprocessing.kbins_discretizer import KBinsDiscretizer


@contextmanager
def does_not_raise():
    yield


class TestKBinsDiscretizer:

    ################# Test for public methods #################
    def test_attributes_to_dict(self):

        discretizer = KBinsDiscretizer()

        discretizer._bin_edges_by_column = {"variable": [0.0, 3.0, 6.0, 9.0]}
        discretizer._bin_labels_by_column = {
            "variable": ["0.0 - 3.0", "3.0 - 6.0", "6.0 - 9.0"]
        }

        actual = discretizer.attributes_to_dict()

        expected = {
            "n_bins": 10,
            "relative_error": 0.01,
            "auto_adapt_bins": False,
            "starting_precision": 0,
            "label_format": "{} - {}",
            "change_endpoint_format": False,
            "_bin_edges_by_column": {"variable": [0.0, 3.0, 6.0, 9.0]},
            "_bin_labels_by_column": {
                "variable": ["0.0 - 3.0", "3.0 - 6.0", "6.0 - 9.0"]
            }
        }

        assert actual == expected

    @pytest.mark.parametrize("attribute",
                             ["n_bins", "relative_error",
                              "auto_adapt_bins", "starting_precision",
                              "label_format", "change_endpoint_format",
                              "_bins_by_column"])
    def test_set_attributes_from_dict(self, attribute):

        discretizer = KBinsDiscretizer()

        params = {
            "n_bins": 5,
            "relative_error": 0.01,
            "auto_adapt_bins": True,
            "starting_precision": 1,
            "label_format": "{} - {}",
            "change_endpoint_format": True,
            "_bin_edges_by_column": {"variable": [0.0, 3.0, 6.0, 9.0]},
            "_bin_labels_by_column": {
                "variable": ["<= 3.0", "3.0 - 6.0", "> 9.0"]
            }
        }

        expected = params[attribute]

        discretizer.set_attributes_from_dict(params)

        actual = getattr(discretizer, attribute)

        assert actual == expected

    ################# Test for private methods #################
    @pytest.mark.parametrize("n_bins, expectation",
                             [(1, pytest.raises(ValueError)),
                              (10.5, pytest.raises(ValueError)),
                              (2, does_not_raise())])
    def test_validate_n_bins_exception(self, n_bins, expectation):
        with expectation:
            assert KBinsDiscretizer()._validate_n_bins(n_bins=n_bins) is None

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

    @pytest.mark.parametrize("change_endpoint_format, bins, expected",
                             [(False, [0, 1, 2, 3],
                               ["0 - 1", "1 - 2", "2 - 3"]),
                              (True, [0, 1, 2, 3],
                               ["<= 1", "1 - 2", "> 2"])],
                             ids=["standard format", "different endpoints"])
    def _create_bin_labels_from_edges(self, change_endpoint_format,
                                      bin_edges, expected):

        discretizer = KBinsDiscretizer(
            change_endpoint_format=change_endpoint_format
        )

        actual = discretizer._create_bin_labels_from_edges(bin_edges)

        assert actual == expected
