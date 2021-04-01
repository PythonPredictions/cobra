from contextlib import contextmanager
import pytest

from typing import Any

import numpy as np
import pandas as pd

from cobra.preprocessing.preprocessor import PreProcessor
from cobra.datasets import make_large_house_prices_dataset


@contextmanager
def does_not_raise():
    yield


class TestPreProcessor:

    @pytest.mark.parametrize("train_prop, selection_prop, validation_prop, "
                             "expected_sizes",
                             [(0.6, 0.2, 0.2, {"train": 6,
                                               "selection": 2,
                                               "validation": 2}),
                              (0.7, 0.3, 0.0, {"train": 7,
                                               "selection": 3}),
                              # Error "The sum of train_prop, selection_prop and
                              # validation_prop must be 1.0." should not be
                              # raised:
                              (0.7, 0.2, 0.1, {"train": 7,
                                               "selection": 2,
                                               "validation": 1})])
    def test_train_selection_validation_split(self, train_prop: float,
                                              selection_prop: float,
                                              validation_prop: float,
                                              expected_sizes: dict):
        X = np.arange(100).reshape(10, 10)
        data = pd.DataFrame(X, columns=[f"c{i+1}" for i in range(10)])
        data.loc[:, "target"] = np.array([0] * 7 + [1] * 3)

        actual = PreProcessor.train_selection_validation_split(data,
                                                               train_prop,
                                                               selection_prop,
                                                               validation_prop)

        # check for the output schema
        assert list(actual.columns) == list(data.columns)

        # check that total size of input & output is the same!
        assert len(actual.index) == len(data.index)

        # check for the output sizes per split
        actual_sizes = actual.groupby("split").size().to_dict()

        assert actual_sizes == expected_sizes

    def test_train_selection_validation_split_error_wrong_prop(self):

        error_msg = ("The sum of train_prop, selection_prop and "
                     "validation_prop must be 1.0.")
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
        with pytest.raises(ValueError, match=error_msg):
            (PreProcessor
             .train_selection_validation_split(df,
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

    @pytest.mark.skip()  # Only meant to be run manually.
    def test_preprocessor_performance_on_large_dataset(self,
                                                       data_folder='../../datasets/argentina-venta-de-propiedades'):
        """
        Download a large housing dataset and run the PreProcessor on it,
        to check and debug its performance.

        This test is meant to be run manually only,
        since it would slow down the automatic tests if included
        *and* there are no real assertions to be made here - it's meant more
        as a thing to run manually and go over (or debug) the code.

        To run:
        - disable the skip marking of this test and debug it
        - call this test from a python console (this will print progress
        messages, the first option won't):
        from
        """
        print("Creating basetable...")
        basetable = make_large_house_prices_dataset(
            data_folder,
            ask_download_confirmation=False)  # input() call doesn't work in pytest.

        # Preparing the preprocessor to do the performance testing:
        preprocessor = PreProcessor.from_params()
        basetable = preprocessor.train_selection_validation_split(basetable,
                                                                  train_prop=0.7,
                                                                  selection_prop=0.15,
                                                                  validation_prop=0.15)

        # Setting which vars are discrete and which are continuous:
        derived_datetime_features = [col for col in basetable.columns
                                     if col.startswith("start_date")
                                     or col.startswith("end_date")
                                     or col.startswith("created_on")]
        hierarchical_location_features = ["l1", "l2", "l3", "l4", "l5", "l6"]
        rooms_features = ["rooms", "bedrooms", "bathrooms"]
        discrete_vars = ["ad_type"] + \
                        derived_datetime_features + \
                        hierarchical_location_features + \
                        rooms_features + \
                        ["property_type", "operation_type", "country"]
        # Note: "title" and "description" are not included here, they would help
        # create a better model, but our primary interest here is testing
        # just the preprocessing performance instead...

        random_feature_cols = [col for col in basetable.columns
                               if col.startswith("random_feature")]
        continuous_vars = ["lat", "lon", "surface_total", "surface_covered"] + \
                          random_feature_cols

        # Setting the target column:
        target_clf = "price_EUR_>300K"
        target_regr = "price_EUR"

        print("Fitting the preprocessor...")
        preprocessor.fit(basetable[basetable["split"] == "train"],
                         continuous_vars=continuous_vars,
                         discrete_vars=discrete_vars,
                         target_column_name=target_clf)

        print("Transforming the preprocessor...")
        basetable = preprocessor.transform(basetable,
                                           continuous_vars=continuous_vars,
                                           discrete_vars=discrete_vars)
