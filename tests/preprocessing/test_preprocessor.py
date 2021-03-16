from contextlib import contextmanager
import pytest

from typing import Any

import numpy as np
import pandas as pd

from cobra.preprocessing.preprocessor import PreProcessor


@contextmanager
def does_not_raise():
    yield


class TestPreProcessor:

    @pytest.mark.parametrize(("train_prop, selection_prop, "
                              "validation_prop, expected_sizes"),
                             [(0.6, 0.2, 0.2, {"train": 6,
                                               "selection": 2,
                                               "validation": 2}),
                              (0.7, 0.3, 0.0, {"train": 7,
                                               "selection": 3})])
    def test_train_selection_validation_split(self, train_prop: float,
                                              selection_prop: float,
                                              validation_prop: float,
                                              expected_sizes: dict):
        X = np.arange(100).reshape(10, 10)
        data = pd.DataFrame(X, columns=[f"c{i+1}" for i in range(10)])
        data.loc[:, "target"] = np.array([0] * 7 + [1] * 3)

        # No stratified split here because sample size is to low to make
        # it work. This feature is already well-tested in scikit-learn and
        # needs no further testing here
        actual = PreProcessor.train_selection_validation_split(data,
                                                               "target",
                                                               train_prop,
                                                               selection_prop,
                                                               validation_prop,
                                                               False)

        # check for the output schema
        expected_schema = list(data.columns) + ["split"]
        assert list(actual.columns) == expected_schema

        # check that total size of input & output is the same!
        assert len(actual.index) == len(data.index)

        # check for the output sizes per split
        actual_sizes = actual.groupby("split").size().to_dict()

        assert actual_sizes == expected_sizes

    def test_train_selection_validation_split_error_wrong_prop(self):

        error_msg = ("The sum of train_prop, selection_prop and "
                     "validation_prop cannot differ from 1.0")
        train_prop = 0.7
        selection_prop = 0.3

        self._test_train_selection_validation_split_error(train_prop,
                                                          selection_prop,
                                                          error_msg)

    def test_train_selection_validation_split_error_zero_selection_prop(self):

        error_msg = "selection_prop cannot be zero!"
        train_prop = 0.9
        selection_prop = 0.0

        self._test_train_selection_validation_split_error(train_prop,
                                                          selection_prop,
                                                          error_msg)

    def _test_train_selection_validation_split_error(self,
                                                     train_prop: float,
                                                     selection_prop: float,
                                                     error_msg: str):
        df = pd.DataFrame()
        cname = ""
        with pytest.raises(ValueError, match=error_msg):
            (PreProcessor
             .train_selection_validation_split(df, cname,
                                               train_prop=train_prop,
                                               selection_prop=selection_prop,
                                               validation_prop=0.1))

    @pytest.mark.parametrize("injection_location, expected",
                             [(None, True),
                              ("categorical_data_processor", False),
                              ("discretizer", False),
                              ("target_encoder", False)])
    def test_is_valid_pipeline(self, injection_location: str,
                               expected: bool):

        # is_valid_pipeline only checks for relevant keys atm
        pipeline_dict = {
            "categorical_data_processor": {
                "regroup": None,
                "regroup_name": None,
                "keep_missing": None,
                "category_size_threshold": None,
                "p_value_threshold": None,
                "scale_contingency_table": None,
                "forced_categories": None,
            },
            "discretizer": {
                "n_bins": None,
                "strategy": None,
                "closed": None,
                "auto_adapt_bins": None,
                "starting_precision": None,
                "label_format": None,
                "change_endpoint_format": None,
            },
            "target_encoder": {
                "weight": None,
                "imputation_strategy": None,
            }
        }

        if injection_location:
            pipeline_dict[injection_location]["wrong_key"] = None

        actual = PreProcessor._is_valid_pipeline(pipeline_dict)

        assert actual == expected

    @pytest.mark.parametrize(("continuous_vars, discrete_vars, expectation, "
                              "expected"),
                             [([], [], pytest.raises(ValueError), None),
                              (["c1", "c2"], ["d1", "d2"], does_not_raise(),
                               ["d1_processed", "d2_processed",
                                "c1_bin", "c2_bin"]),
                              (["c1", "c2"], [], does_not_raise(),
                               ["c1_bin", "c2_bin"]),
                              ([], ["d1", "d2"], does_not_raise(),
                               ["d1_processed", "d2_processed"])])
    def test_get_variable_list(self, continuous_vars: list,
                               discrete_vars: list,
                               expectation: Any,
                               expected: list):

        with expectation:
            actual = PreProcessor._get_variable_list(continuous_vars,
                                                     discrete_vars)

            assert actual == expected

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

        categorical_data_processor = CategoricalDataProcessor(
                    category_size_threshold=0,
                    p_value_threshold=0.0001)

        categorical_data_processor.fit(data,
                                       discrete_vars,
                                       target_column_name)

        actual = categorical_data_processor.transform(data,
                                                      discrete_vars)

        pd.testing.assert_frame_equal(actual, expected)
