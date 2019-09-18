import pandas as pd
import pytest

from cobra.preprocessing.kbins_discretizer import KBinsDiscretizer


class TestKBinsDiscretizer:

    # tests for _validate_n_bins function

    def test_validate_n_bins_exception_1(self):

        with pytest.raises(ValueError):
            KBinsDiscretizer()._validate_n_bins(n_bins=1)

    def test_validate_n_bins_exception_no_integral(self):

        with pytest.raises(ValueError):
            KBinsDiscretizer()._validate_n_bins(n_bins=10.5)

    def test_validate_n_bins_valid_n_bins(self):

        KBinsDiscretizer()._validate_n_bins(n_bins=2)

    # tests for _compute_minimal_precision_of_cutpoints

    def test_compute_minimal_precision_of_cutpoints_less_precision(self):
        # If starting precision is bigger than actual precision, should return
        # starting precision

        cutpoints = [-10, 0, 1, 2]
        discretizer = KBinsDiscretizer(starting_precision=1)
        res = discretizer._compute_minimal_precision_of_cutpoints(cutpoints)
        assert res == 1

    def test_compute_minimal_precision_of_cutpoints_more_precision(self):
        # If starting precision is smaller than actual precision, should return
        # actual precision

        cutpoints = [-10, 0, 1, 1.01]
        discretizer = KBinsDiscretizer()
        res = discretizer._compute_minimal_precision_of_cutpoints(cutpoints)
        assert res == 2

    def test_compute_minimal_precision_of_cutpoints_equal_precision(self):
        # If starting precision is equal to actual precision, should return
        # starting precision

        cutpoints = [-10, 0, 1, 1.1]
        discretizer = KBinsDiscretizer(starting_precision=1)
        res = discretizer._compute_minimal_precision_of_cutpoints(cutpoints)
        assert res == 1

    def test_compute_minimal_precision_of_cutpoints_negative_start(self):
        # Check if negative starting precision also leads to the correct result

        cutpoints = [-10, 0, 1, 2]
        discretizer = KBinsDiscretizer(starting_precision=-1)
        res = discretizer._compute_minimal_precision_of_cutpoints(cutpoints)
        assert res == 0

    def test_compute_minimal_precision_of_cutpoints_round_up(self):
        # Check if negative starting precision leads to rounding up
        # bin edges to the nearest multiple of 10

        cutpoints = [-10, 0, 10, 21]
        discretizer = KBinsDiscretizer(starting_precision=-1)
        res = discretizer._compute_minimal_precision_of_cutpoints(cutpoints)
        assert res == -1

    # tests for _compute_bins_from_cutpoints

    # tests for _create_bin_labels
