import pytest

import numpy as np
import pandas as pd

from cobra.preprocessing import CategoricalDataProcessor


class TestCategoricalDataProcessor:

    @pytest.mark.parametrize("scale_contingency_table, expected",
                             [(False, 0.013288667),
                              (True, 0.434373)])
    def test_compute_p_value(self, scale_contingency_table, expected):

        X = pd.Series(data=(["c1"]*70 + ["c2"]*20 + ["c3"]*10))
        y = pd.Series(data=([0]*35 + [1]*35 + [0]*15 + [1]*5 + [0]*8 + [1]*2))
        category = "c1"

        actual = (CategoricalDataProcessor
                  ._compute_p_value(X, y, category, scale_contingency_table))

        assert pytest.approx(actual) == expected

    def test_get_small_categories(self):

        data = pd.Series(data=(["c1"]*50 + ["c2"]*25 + ["c3"]*15 + ["c4"]*5))
        incidence = 0.35
        threshold = 10  # to make it easy to manualy compute
        expected = {"c3", "c4"}

        actual = (CategoricalDataProcessor
                  ._get_small_categories(data, incidence, threshold))

        assert actual == expected

    def test_replace_missings(self):

        data = pd.DataFrame({"variable": ["c1", "c2", np.nan, "", " "]})
        expected = pd.DataFrame({"variable": ["c1", "c2", "Missing", "Missing",
                                              "Missing"]
                                 })
        actual = (CategoricalDataProcessor
                  ._replace_missings(data, ["variable"]))

        pd.testing.assert_frame_equal(actual, expected)

    @pytest.mark.parametrize("combined_categories, expected",
                             [({"c3", "c4"},
                               pd.Series(data=["c1", "c2", "Other", "Other"])),
                              ({}, pd.Series(data=["c1", "c2", "c3", "c4"]))])
    def test_replace_categories(self, combined_categories, expected):

        data = pd.Series(data=["c1", "c2", "c3", "c4"])

        actual = (CategoricalDataProcessor
                  ._replace_categories(data, combined_categories))

        pd.testing.assert_series_equal(actual, expected)
