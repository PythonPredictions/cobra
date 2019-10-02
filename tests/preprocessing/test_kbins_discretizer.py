import pandas as pd
import pytest

from cobra.preprocessing.kbins_discretizer import KBinsDiscretizer


class TestKBinsDiscretizer:

    ################# Test for public methods #################

    ################# Test for private methods #################
    # Tests for _validate_n_bins function
    def test_kbins_discretizer_validate_n_bins_exception_1(self):

        with pytest.raises(ValueError):
            KBinsDiscretizer()._validate_n_bins(n_bins=1)

    def test_kbins_discretizer_validate_n_bins_exception_no_integral(self):

        with pytest.raises(ValueError):
            KBinsDiscretizer()._validate_n_bins(n_bins=10.5)

    def test_kbins_discretizer_validate_n_bins_valid_n_bins(self):

        KBinsDiscretizer()._validate_n_bins(n_bins=2)

    # Test for _compute_bin_edges
    def test_kbins_discretizer_compute_bin_edges_quantile_method(self):

        data = pd.DataFrame({"variable": list(range(0, 11))})  # ints from 0-10

        discretizer = KBinsDiscretizer()
        actual = discretizer._compute_bin_edges(data, column_name="variable",
                                                n_bins=4,
                                                col_min=data.variable.min(),
                                                col_max=data.variable.max())
        expected = [0.0, 2.5, 5, 7.5, 10.0]

        assert expected == actual

    def test_kbins_discretizer_compute_bin_edges_uniform_method(self):

        data = pd.DataFrame({"variable": list(range(0, 10))})  # ints from 0-9

        discretizer = KBinsDiscretizer(strategy="uniform")
        actual = discretizer._compute_bin_edges(data, column_name="variable",
                                                n_bins=3,
                                                col_min=data.variable.min(),
                                                col_max=data.variable.max())
        expected = [0.0, 3.0, 6.0, 9.0]

        assert expected == actual

    # Tests for _compute_minimal_precision_of_bin_edges
    def test_compute_minimal_precision_of_bin_edges_less_precision(self):
        # If starting precision is bigger than actual precision, should return
        # starting precision

        bin_edges = [-10, 0, 1, 2]
        discretizer = KBinsDiscretizer(starting_precision=1)
        res = discretizer._compute_minimal_precision_of_bin_edges(bin_edges)
        assert res == 1

    def test_compute_minimal_precision_of_bin_edges_more_precision(self):
        # If starting precision is smaller than actual precision, should return
        # actual precision

        bin_edges = [-10, 0, 1, 1.01]
        discretizer = KBinsDiscretizer()
        res = discretizer._compute_minimal_precision_of_bin_edges(bin_edges)
        assert res == 2

    def test_compute_minimal_precision_of_bin_edges_equal_precision(self):
        # If starting precision is equal to actual precision, should return
        # starting precision

        bin_edges = [-10, 0, 1, 1.1]
        discretizer = KBinsDiscretizer(starting_precision=1)
        res = discretizer._compute_minimal_precision_of_bin_edges(bin_edges)
        assert res == 1

    def test_compute_minimal_precision_of_bin_edges_negative_start(self):
        # Check if negative starting precision also leads to the correct result

        bin_edges = [-10, 0, 1, 2]
        discretizer = KBinsDiscretizer(starting_precision=-1)
        res = discretizer._compute_minimal_precision_of_bin_edges(bin_edges)
        assert res == 0

    def test_compute_minimal_precision_of_bin_edges_round_up(self):
        # Check if negative starting precision leads to rounding up
        # bin edges to the nearest multiple of 10

        bin_edges = [-10, 0, 10, 21]
        discretizer = KBinsDiscretizer(starting_precision=-1)
        res = discretizer._compute_minimal_precision_of_bin_edges(bin_edges)
        assert res == -1

    # Tests for _compute_bins_from_edges
    def test_kbins_discretizer_compute_bins_from_edges(self):

        bin_edges = [0, 1, 1.5, 2]

        discretizer = KBinsDiscretizer()
        actual = discretizer._compute_bins_from_edges(bin_edges)

        expected = [(0, 1), (1, 1.5), (1.5, 2)]
        assert actual == expected

    def test_kbins_discretizer_compute_bins_from_edges_round_up(self):

        bin_edges = [0, 1, 1.5, 3]

        discretizer = KBinsDiscretizer()
        actual = discretizer._compute_bins_from_edges(bin_edges)

        expected = [(0, 1), (1, 2), (2, 3)]
        assert actual == expected

    # Tests for _create_bin_labels
    def test_kbins_discretizer_create_bin_labels(self):

        bins = [(0, 1), (1, 2), (2, 3)]

        discretizer = KBinsDiscretizer()
        actual = discretizer._create_bin_labels(bins)
        expected = ["0 - 1", "1 - 2", "2 - 3"]

        assert actual == expected

    def test_kbins_discretizer_create_bin_labels_different_endpoint_fmt(self):

        bins = [(0, 1), (1, 2), (2, 3)]

        discretizer = KBinsDiscretizer(change_endpoint_format=True)
        actual = discretizer._create_bin_labels(bins)
        expected = ["< 1", "1 - 2", "> 2"]

        assert actual == expected
