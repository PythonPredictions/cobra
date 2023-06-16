from contextlib import contextmanager
from typing import Any
from unittest.mock import MagicMock
import pytest
import numpy as np
import pandas as pd
from pytest_mock import MockerFixture

from cobra.preprocessing.preprocessor import PreProcessor


@contextmanager
def does_not_raise():
    yield


class TestPreProcessor:
    @pytest.mark.parametrize(
        "train_prop, selection_prop, validation_prop, " "expected_sizes",
        [
            (0.6, 0.2, 0.2, {"train": 6, "selection": 2, "validation": 2}),
            (0.7, 0.3, 0.0, {"train": 7, "selection": 3}),
            # Error "The sum of train_prop, selection_prop and
            # validation_prop must be 1.0." should not be
            # raised:
            (0.7, 0.2, 0.1, {"train": 7, "selection": 2, "validation": 1}),
        ],
    )
    def test_train_selection_validation_split(
        self,
        train_prop: float,
        selection_prop: float,
        validation_prop: float,
        expected_sizes: dict,
    ):
        X = np.arange(100).reshape(10, 10)
        data = pd.DataFrame(X, columns=[f"c{i+1}" for i in range(10)])
        data.loc[:, "target"] = np.array([0] * 7 + [1] * 3)

        actual = PreProcessor.train_selection_validation_split(
            data, train_prop, selection_prop, validation_prop
        )

        # check for the output schema
        assert list(actual.columns) == list(data.columns)

        # check that total size of input & output is the same!
        assert len(actual.index) == len(data.index)

        # check for the output sizes per split
        actual_sizes = actual.groupby("split").size().to_dict()

        assert actual_sizes == expected_sizes

    def test_train_selection_validation_split_error_wrong_prop(self):

        error_msg = (
            "The sum of train_prop, selection_prop and " "validation_prop must be 1.0."
        )
        train_prop = 0.7
        selection_prop = 0.3

        self._test_train_selection_validation_split_error(
            train_prop, selection_prop, error_msg
        )

    def test_train_selection_validation_split_error_zero_selection_prop(self):

        error_msg = "selection_prop cannot be zero!"
        train_prop = 0.9
        selection_prop = 0.0

        self._test_train_selection_validation_split_error(
            train_prop, selection_prop, error_msg
        )

    def _test_train_selection_validation_split_error(
        self, train_prop: float, selection_prop: float, error_msg: str
    ):
        df = pd.DataFrame()
        with pytest.raises(ValueError, match=error_msg):
            (
                PreProcessor.train_selection_validation_split(
                    df,
                    train_prop=train_prop,
                    selection_prop=selection_prop,
                    validation_prop=0.1,
                )
            )

    @pytest.mark.parametrize(
        "injection_location, expected",
        [
            (None, True),
            ("categorical_data_processor", False),
            ("discretizer", False),
            ("target_encoder", False),
        ],
    )
    def test_is_valid_pipeline(self, injection_location: str, expected: bool):

        # is_valid_pipeline only checks for relevant keys atm
        pipeline_dict = {
            "categorical_data_processor": {
                "model_type": None,
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
            },
        }

        if injection_location:
            pipeline_dict[injection_location]["wrong_key"] = None

        actual = PreProcessor._is_valid_pipeline(pipeline_dict)

        assert actual == expected

    @pytest.mark.parametrize(
        ("continuous_vars, discrete_vars, expectation, " "expected"),
        [
            ([], [], pytest.raises(ValueError), None),
            (
                ["c1", "c2"],
                ["d1", "d2"],
                does_not_raise(),
                ["d1_processed", "d2_processed", "c1_bin", "c2_bin"],
            ),
            (["c1", "c2"], [], does_not_raise(), ["c1_bin", "c2_bin"]),
            ([], ["d1", "d2"], does_not_raise(), ["d1_processed", "d2_processed"]),
        ],
    )
    def test_get_variable_list(
        self,
        continuous_vars: list,
        discrete_vars: list,
        expectation: Any,
        expected: list,
    ):

        with expectation:
            actual = PreProcessor._get_variable_list(continuous_vars, discrete_vars)

            assert actual == expected

    @pytest.mark.parametrize(
    ("input, expected"),
    [
        # example 1
        (
            pd.DataFrame({
                "ID": list(range(20)),
                "A": [1,2,3,4,5,6,7,8,9,9,8,9,8,9,6,5,6,6,9,8],
                "B": ["Cat"] *5 + ["Dog"]*10 + ["Fish"]*5,
                "C": [1,2,3,4,9,10,11,12,13,5,6,7,8,15,19,18,14,16,13,17],
                "Target": [1]*2 + [0]*5 + [1]*3 + [0]*5 + [1]*5
                }
            ),
            pd.DataFrame({
                'ID': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
                'A': [1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 8, 9, 8, 9, 6, 5, 6, 6, 9, 8],
                'B': ['Cat','Cat','Cat','Cat','Cat','Dog','Dog','Dog','Dog','Dog','Dog','Dog','Dog','Dog','Dog','Fish','Fish','Fish','Fish','Fish'],
                'C': [1, 2, 3, 4, 9, 10, 11, 12, 13, 5, 6, 7, 8, 15, 19, 18, 14, 16, 13, 17],
                'Target': [1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
                'C_bin': ['1.0 - 3.0','1.0 - 3.0','1.0 - 3.0','3.0 - 5.0','7.0 - 9.0','9.0 - 10.0','10.0 - 12.0','10.0 - 12.0','12.0 - 13.0','3.0 - 5.0','5.0 - 7.0','5.0 - 7.0','7.0 - 9.0','13.0 - 15.0','17.0 - 19.0','17.0 - 19.0','13.0 - 15.0','15.0 - 17.0','12.0 - 13.0','15.0 - 17.0'],
                'B_processed': ['Cat','Cat','Cat','Cat','Cat','Dog','Dog','Dog','Dog','Dog','Dog','Dog','Dog','Dog','Dog','Fish','Fish','Fish','Fish','Fish'],
                'A_processed': [1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 8, 9, 8, 9, 6, 5, 6, 6, 9, 8],
                'B_enc': [0.4,0.4,0.4,0.4,0.4,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,1.0,1.0,1.0,1.0,1.0],
                'A_enc': [1.0,1.0,0.0,0.0,0.5,0.5,0.0,0.5,0.6,0.6,0.5,0.6,0.5,0.6,0.5,0.5,0.5,0.5,0.6,0.5],
                'C_enc': [0.6666666666666666,0.6666666666666666,0.6666666666666666,0.5,0.0,0.0,0.5,0.5,1.0,0.5,0.0,0.0,0.0,0.5,0.5,0.5,0.5,1.0,1.0,1.0]
                }
            ),
        )
    ]
    )
    def test_fit_transform_without_id_col_name(self, input, expected):
        
        preprocessor = PreProcessor.from_params(model_type="classification")
        
        continuous_vars, discrete_vars = preprocessor.get_continuous_and_discrete_columns(input, "ID","Target")

        calculated = preprocessor.fit_transform(
            input,
            continuous_vars=continuous_vars,
            discrete_vars=discrete_vars,
            target_column_name="Target"
            )
        pd.testing.assert_frame_equal(calculated, expected, check_dtype=False, check_categorical=False)

    @pytest.mark.parametrize(
    ("input, expected"),
    [
        # example 1
        (
            pd.DataFrame({
                "ID": list(range(20)),
                "A": [1,2,3,4,5,6,7,8,9,9,8,9,8,9,6,5,6,6,9,8],
                "B": ["Cat"] *5 + ["Dog"]*10 + ["Fish"]*5,
                "C": [1,2,3,4,9,10,11,12,13,5,6,7,8,15,19,18,14,16,13,17],
                "Target": [1]*2 + [0]*5 + [1]*3 + [0]*5 + [1]*5
                }
            ),
            pd.DataFrame({
                'ID': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
                'A': [1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 8, 9, 8, 9, 6, 5, 6, 6, 9, 8],
                'B': ['Cat','Cat','Cat','Cat','Cat','Dog','Dog','Dog','Dog','Dog','Dog','Dog','Dog','Dog','Dog','Fish','Fish','Fish','Fish','Fish'],
                'C': [1, 2, 3, 4, 9, 10, 11, 12, 13, 5, 6, 7, 8, 15, 19, 18, 14, 16, 13, 17],
                'Target': [1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
                'C_bin': ['1.0 - 3.0','1.0 - 3.0','1.0 - 3.0','3.0 - 5.0','7.0 - 9.0','9.0 - 10.0','10.0 - 12.0','10.0 - 12.0','12.0 - 13.0','3.0 - 5.0','5.0 - 7.0','5.0 - 7.0','7.0 - 9.0','13.0 - 15.0','17.0 - 19.0','17.0 - 19.0','13.0 - 15.0','15.0 - 17.0','12.0 - 13.0','15.0 - 17.0'],
                'B_processed': ['Cat','Cat','Cat','Cat','Cat','Dog','Dog','Dog','Dog','Dog','Dog','Dog','Dog','Dog','Dog','Fish','Fish','Fish','Fish','Fish'],
                'A_processed': [1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 8, 9, 8, 9, 6, 5, 6, 6, 9, 8],
                'B_enc': [0.4,0.4,0.4,0.4,0.4,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,1.0,1.0,1.0,1.0,1.0],
                'A_enc': [1.0,1.0,0.0,0.0,0.5,0.5,0.0,0.5,0.6,0.6,0.5,0.6,0.5,0.6,0.5,0.5,0.5,0.5,0.6,0.5],
                'C_enc': [0.6666666666666666,0.6666666666666666,0.6666666666666666,0.5,0.0,0.0,0.5,0.5,1.0,0.5,0.0,0.0,0.0,0.5,0.5,0.5,0.5,1.0,1.0,1.0]
                }
            ),
        )
    ]
    )
    def test_fit_transform_with_id_col_name(self, input, expected):
        
        preprocessor = PreProcessor.from_params(model_type="classification")
        
        # continuous_vars, discrete_vars = preprocessor.get_continous_and_discreate_columns(input, "ID","Target")

        calculated = preprocessor.fit_transform(
            input,
            continuous_vars=None,
            discrete_vars=None,
            target_column_name="Target",
            id_col_name="ID"
            )
        pd.testing.assert_frame_equal(calculated, expected, check_dtype=False, check_categorical=False)

    @staticmethod
    def mock_transform(df: pd.DataFrame, args):
        """Mock the transform method."""
        df["new_column"] = "Hello World"
        return df

    def test_mutable_train_data_fit_transform(self, mocker: MockerFixture):
        """Test if the train_data input is not changed when performing fit_transform."""
        train_data = pd.DataFrame(
            [[1, "2", 3], [10, "20", 30], [100, "200", 300]],
            columns=["foo", "bar", "baz"],
        )
        preprocessor = PreProcessor.from_params(
            model_type="classification", n_bins=10, weight=0.8
        )
        preprocessor._categorical_data_processor = MagicMock()
        preprocessor._categorical_data_processor.transform = self.mock_transform
        preprocessor._discretizer = MagicMock()
        preprocessor._discretizer.transform = self.mock_transform
        preprocessor._target_encoder = MagicMock()
        preprocessor._target_encoder.transform = self.mock_transform

        result = preprocessor.fit_transform(
            train_data,
            continuous_vars=["foo"],
            discrete_vars=["bar"],
            target_column_name=["baz"],
        )
        assert "new_column" not in train_data.columns
        assert "new_column" in result.columns

    @pytest.mark.parametrize(
        ("input, expected"),
        [
            # example 1
            (
                pd.DataFrame(
                    {
                        "a": [1, 8, np.nan],
                        "b": [np.nan, 8, np.nan],
                        "c": [np.nan, np.nan, np.nan],
                        "d": [np.nan, np.nan, 5],
                        "e": [1, 960, np.nan],
                        "f": [np.nan, np.nan, np.nan],
                    }
                ),
                pd.DataFrame(
                    {
                        "a": [1.0, 8.0, np.nan],
                        "b": [np.nan, 8.0, np.nan],
                        "d": [np.nan, np.nan, 5.0],
                        "e": [1.0, 960.0, np.nan],
                    }
                ),
            ),
            # example 2
            (
                pd.DataFrame(
                    {
                        "a": [1, 8, np.nan],
                        "b": [np.nan, 8, np.nan],
                        "c": [np.nan, np.nan, np.nan],
                        "d": [np.nan, np.nan, 5],
                        "e": [1, 960, np.nan],
                    }
                ),
                pd.DataFrame(
                    {
                        "a": [1.0, 8.0, np.nan],
                        "b": [np.nan, 8.0, np.nan],
                        "d": [np.nan, np.nan, 5.0],
                        "e": [1.0, 960.0, np.nan],
                    }
                ),
            ),
            # example 3
            (
                pd.DataFrame(
                    {
                        "a": [1, 8, np.nan],
                        "b": [np.nan, 8, np.nan],
                        "d": [np.nan, np.nan, 5],
                        "e": [1, 960, np.nan],
                    }
                ),
                pd.DataFrame(
                    {
                        "a": [1.0, 8.0, np.nan],
                        "b": [np.nan, 8.0, np.nan],
                        "d": [np.nan, np.nan, 5.0],
                        "e": [1.0, 960.0, np.nan],
                    }
                ),
            ),
            # example 4 categorical
            (
                pd.DataFrame(
                    {
                        "a": [1, 8, np.nan],
                        "b": [np.nan, np.nan, np.nan],
                        "d": [np.nan, np.nan, 5],
                        "e": [1, 960, np.nan],
                        "category_1": ["A", "A", "B"],
                        "category_2": [np.nan, "A", "B"],
                        "category_3": [np.nan, np.nan, np.nan],
                    },
                ).astype(
                    {
                        "a": np.float64(),
                        "b": np.float64(),
                        "d": np.float64(),
                        "e": np.float64(),
                        "category_1": pd.CategoricalDtype(),
                        "category_2": pd.CategoricalDtype(),
                        "category_3": pd.CategoricalDtype(),
                    }
                ),
                pd.DataFrame(
                    {
                        "a": [1, 8, np.nan],
                        "d": [np.nan, np.nan, 5],
                        "e": [1, 960, np.nan],
                        "category_1": ["A", "A", "B"],
                        "category_2": [np.nan, "A", "B"],
                    }
                ).astype(
                    {
                        "a": np.float64(),
                        "d": np.float64(),
                        "e": np.float64(),
                        "category_1": pd.CategoricalDtype(),
                        "category_2": pd.CategoricalDtype(),
                    }
                ),
            ),
        ],
    )
    def test_drops_columns_containing_only_nan(self, input, expected):

        print(input)
        output = PreProcessor._check_nan_columns_and_drop_columns_containing_only_nan(
            input
        )

        print(output)
        print(expected)
        assert output.equals(expected)
