import pytest

import numpy as np
import pandas as pd

from cobra.preprocessing import CategoricalDataProcessor


class TestCategoricalDataProcessor:

    def test_attributes_to_dict(self):

        processor = CategoricalDataProcessor()

        cleaned_categories = ["a", "b", "c"]
        processor._cleaned_categories_by_column = {
            "variable": set(cleaned_categories)
        }

        actual = processor.attributes_to_dict()

        expected = {
            "model_type": "classification",
            "regroup": True,
            "regroup_name": "Other",
            "keep_missing": True,
            "category_size_threshold": 5,
            "p_value_threshold": 0.001,
            "scale_contingency_table": True,
            "forced_categories": {},
            "_cleaned_categories_by_column": {
                "variable": list(set(cleaned_categories))
            }
        }

        assert actual == expected

    @pytest.mark.parametrize("attribute",
                             ["regroup", "regroup_name", "keep_missing",
                              "category_size_threshold", "p_value_threshold",
                              "scale_contingency_table", "forced_categories",
                              "_cleaned_categories_by_column"])
    def test_set_attributes_from_dict(self, attribute):

        processor = CategoricalDataProcessor()

        cleaned_categories = ["a", "b", "c"]
        params = {
            "regroup": True,
            "regroup_name": "Other",
            "keep_missing": True,
            "category_size_threshold": 5,
            "p_value_threshold": 0.001,
            "scale_contingency_table": True,
            "forced_categories": {},
            "_cleaned_categories_by_column": {
                "variable": cleaned_categories
            }
        }

        expected = params[attribute]

        if attribute == "_cleaned_categories_by_column":
            # list is transformed to a set in CategoricalDataProcessor
            expected = {"variable": set(cleaned_categories)}

        processor.set_attributes_from_dict(params)

        actual = getattr(processor, attribute)

        assert actual == expected

    @pytest.mark.parametrize("scale_contingency_table, expected",
                             [(False, 0.01329),
                              (True, 0.43437)])
    def test_compute_p_value_classification(self, scale_contingency_table, expected):

        X = pd.Series(data=(["c1"]*70 + ["c2"]*20 + ["c3"]*10))
        y = pd.Series(data=([0]*35 + [1]*35 + [0]*15 + [1]*5 + [0]*8 + [1]*2))
        category = "c1"

        actual = (CategoricalDataProcessor
                  ._compute_p_value(X, y, category, "classification", scale_contingency_table))

        assert pytest.approx(actual, abs=1e-5) == expected

    @pytest.mark.parametrize("seed, expected",
                             [(505, 0.02222),
                              (603, 0.89230)])
    def test_compute_p_value_regression(self, seed, expected):

        np.random.seed(seed)

        X = pd.Series(data=(["c1"]*70 + ["c2"]*20 + ["c3"]*10))
        y = pd.Series(data=np.random.uniform(0, 1, 100)*5)
        category = "c1"

        actual = (CategoricalDataProcessor
                  ._compute_p_value(X, y, category, "regression", None))

        assert pytest.approx(actual, abs=1e-5) == expected

    def test_get_small_categories(self):

        data = pd.Series(data=(["c1"]*50 + ["c2"]*25 + ["c3"]*15 + ["c4"]*5))
        incidence = 0.35
        threshold = 10  # to make it easy to manualLy compute
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

    @pytest.mark.parametrize("cleaned_categories, expected",
                             [({"c1", "c2"},
                               pd.Series(data=["c1", "c2", "Other", "Other"])),
                              ({"c1", "c2", "c3", "c4"},
                               pd.Series(data=["c1", "c2", "c3", "c4"]))])
    def test_replace_categories(self, cleaned_categories, expected):

        data = pd.Series(data=["c1", "c2", "c3", "c4"])

        actual = (CategoricalDataProcessor
                  ._replace_categories(data, cleaned_categories, 'Other'))

        pd.testing.assert_series_equal(actual, expected)

    def test_all_cats_not_significant(self):
        # Expected
        e = {'categorical_var': ['A', 'A', 'A', 'A',
                                 'B', 'B', 'B', 'B',
                                 'C', 'C', 'C', 'C'],
             'target': [1, 1, 1, 1,
                        0, 0, 0, 0,
                        1, 0, 1, 0],
             'categorical_var_processed': ['A', 'A', 'A', 'A',
                                           'B', 'B', 'B', 'B',
                                           'C', 'C', 'C', 'C']}

        # data -> actual
        d = {'categorical_var': ['A', 'A', 'A', 'A',
                                 'B', 'B', 'B', 'B',
                                 'C', 'C', 'C', 'C'],
             'target': [1, 1, 1, 1,
                        0, 0, 0, 0,
                        1, 0, 1, 0]}

        discrete_vars = ['categorical_var']
        target_column_name = 'target'

        data = pd.DataFrame(d, columns=['categorical_var', 'target'])
        expected = pd.DataFrame(e, columns=['categorical_var',
                                            'target',
                                            'categorical_var_processed'])

        categorical_data_processor = CategoricalDataProcessor(
                    category_size_threshold=0,
                    p_value_threshold=0.0001)

        categorical_data_processor.fit(data,
                                       discrete_vars,
                                       target_column_name)

        actual = categorical_data_processor.transform(data,
                                                      discrete_vars)

        pd.testing.assert_frame_equal(actual, expected)

    def test_regroup_name(self):
        # Expected
        e = {'categorical_var': ['A', 'A', 'A', 'A', 'A', 'A',
                                 'B', 'B', 'B', 'B', 'B', 'B',
                                 'C', 'C', 'C', 'C', 'C', 'C'],
             'target': [1, 1, 1, 1, 1, 1,
                        0, 0, 0, 0, 0, 0,
                        1, 0, 1, 0, 1, 0],
             'categorical_var_processed': [
                'A', 'A', 'A', 'A', 'A', 'A',
                'B', 'B', 'B', 'B', 'B', 'B',
                'OTH', 'OTH', 'OTH', 'OTH', 'OTH', 'OTH']}

        # data -> actual
        d = {'categorical_var': ['A', 'A', 'A', 'A', 'A', 'A',
                                 'B', 'B', 'B', 'B', 'B', 'B',
                                 'C', 'C', 'C', 'C', 'C', 'C'],
             'target': [1, 1, 1, 1, 1, 1,
                        0, 0, 0, 0, 0, 0,
                        1, 0, 1, 0, 1, 0]}

        discrete_vars = ['categorical_var']
        target_column_name = 'target'

        data = pd.DataFrame(d, columns=['categorical_var', 'target'])
        expected = pd.DataFrame(e, columns=['categorical_var',
                                            'target',
                                            'categorical_var_processed'])

        expected['categorical_var_processed'] = (
            expected['categorical_var_processed'].astype("category"))

        categorical_data_processor = CategoricalDataProcessor(
                    category_size_threshold=0,
                    regroup_name='OTH',
                    p_value_threshold=0.05)

        categorical_data_processor.fit(data,
                                       discrete_vars,
                                       target_column_name)

        actual = categorical_data_processor.transform(data,
                                                      discrete_vars)

        pd.testing.assert_frame_equal(actual, expected)

    def test_force_category(self):
        # Expected
        e = {'categorical_var': ['A', 'A', 'A', 'A', 'A', 'A',
                                 'B', 'B', 'B', 'B', 'B', 'B',
                                 'C', 'C', 'C', 'C', 'C', 'C'],
             'target': [1, 1, 1, 1, 1, 1,
                        0, 0, 0, 0, 0, 0,
                        1, 0, 1, 0, 1, 0],
             'categorical_var_processed': ['A', 'A', 'A', 'A', 'A', 'A',
                                           'B', 'B', 'B', 'B', 'B', 'B',
                                           'C', 'C', 'C', 'C', 'C', 'C']}

        # data -> actual
        d = {'categorical_var': ['A', 'A', 'A', 'A', 'A', 'A',
                                 'B', 'B', 'B', 'B', 'B', 'B',
                                 'C', 'C', 'C', 'C', 'C', 'C'],
             'target': [1, 1, 1, 1, 1, 1,
                        0, 0, 0, 0, 0, 0,
                        1, 0, 1, 0, 1, 0]}

        discrete_vars = ['categorical_var']
        target_column_name = 'target'

        data = pd.DataFrame(d, columns=['categorical_var', 'target'])
        expected = pd.DataFrame(e, columns=['categorical_var',
                                            'target',
                                            'categorical_var_processed'])

        expected['categorical_var_processed'] = (
            expected['categorical_var_processed'].astype("category"))

        categorical_data_processor = CategoricalDataProcessor(
                    category_size_threshold=0,
                    forced_categories={'categorical_var': ['C']},
                    p_value_threshold=0.05)

        categorical_data_processor.fit(data,
                                       discrete_vars,
                                       target_column_name)

        actual = categorical_data_processor.transform(data,
                                                      discrete_vars)

        pd.testing.assert_frame_equal(actual, expected)

    def test_categorical_variable_is_constant(self):
        # Expected
        e = {'categorical_var': ['A', 'A', 'A', 'A',
                                 'A', 'A', 'A', 'A',
                                 'A', 'A', 'A', 'A'],
             'target': [1, 1, 1, 1,
                        0, 0, 0, 0,
                        1, 0, 1, 0],
             'categorical_var_processed': ['A', 'A', 'A', 'A',
                                           'A', 'A', 'A', 'A',
                                           'A', 'A', 'A', 'A']}

        # data -> actual
        d = {'categorical_var': ['A', 'A', 'A', 'A',
                                 'A', 'A', 'A', 'A',
                                 'A', 'A', 'A', 'A'],
             'target': [1, 1, 1, 1,
                        0, 0, 0, 0,
                        1, 0, 1, 0]}

        discrete_vars = ['categorical_var']
        target_column_name = 'target'

        data = pd.DataFrame(d, columns=['categorical_var', 'target'])
        expected = pd.DataFrame(e, columns=['categorical_var',
                                            'target',
                                            'categorical_var_processed'])

        expected['categorical_var_processed'] = (
            expected['categorical_var_processed'].astype("category"))

        categorical_data_processor = CategoricalDataProcessor(
                    category_size_threshold=0,
                    p_value_threshold=0.0001)

        categorical_data_processor.fit(data,
                                       discrete_vars,
                                       target_column_name)

        actual = categorical_data_processor.transform(data,
                                                      discrete_vars)

        pd.testing.assert_frame_equal(actual, expected)
