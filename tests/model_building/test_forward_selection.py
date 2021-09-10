from contextlib import contextmanager
import pytest

import pandas as pd

from cobra.model_building.models import LogisticRegressionModel, LinearRegressionModel
from cobra.model_building.forward_selection import ForwardFeatureSelection


@contextmanager
def does_not_raise():
    yield

def mock_data(add_split_col: bool=False, model_type="classification"):
    data = pd.DataFrame({"var1_enc": [0.42] * 10,
                         "var2_enc": [0.94] * 10,
                         "var3_enc": [0.87] * 10})

    if model_type == "classification":
        data["target"] = ([0] * 5 + [1] * 2 + [0] * 2 + [1])
    elif model_type == "regression":
        data["target"] = [7, 2, 2, 9, 7, 3, 1, 4, 8, 5]

    if add_split_col:
        data.loc[:, "split"] = (["train"] * 7 + ["selection"] * 3)

    return data

def mock_model_num_pred(n_predictors, model_type="classification"):
    predictors = [f"var{i + 1}_enc" for i in range(n_predictors)]
    return mock_model(predictors, model_type)

def mock_model(predictor_list, model_type="classification"):
    if model_type == "classification":
        model = LogisticRegressionModel()
    elif model_type == "regression":
        model = LinearRegressionModel()

    model.predictors = predictor_list

    return model


class TestForwardFeatureSelection:

    def test_get_model_from_step(self):

        forward_selection = ForwardFeatureSelection()

        with pytest.raises(ValueError):
            forward_selection.get_model_from_step(2)

    @pytest.mark.parametrize("model_type", ["classification", "regression"])
    def test_compute_model_performances(self, mocker, model_type):

        data = mock_data(add_split_col=True, model_type=model_type)

        fw_selection = ForwardFeatureSelection(model_type=model_type)
        fw_selection._fitted_models = [
            mock_model_num_pred(1, model_type=model_type),
            mock_model_num_pred(2, model_type=model_type),
            mock_model_num_pred(3, model_type=model_type)
        ]

        def mock_evaluate(self, X, y, split):  # on AUC scale, but gives the same for RMSE as it is a mock
            if split == "train":
                return 0.612
            else:
                return 0.609

        if model_type == "classification":
            patch_fct = "cobra.model_building.forward_selection.LogisticRegressionModel.evaluate"
        elif model_type == "regression":
            patch_fct = "cobra.model_building.forward_selection.LinearRegressionModel.evaluate"

        mocker.patch(patch_fct, mock_evaluate)

        actual = (fw_selection
                  .compute_model_performances(data, "target",
                                              splits=["train", "selection"]))

        expected = pd.DataFrame([
            {"predictors": ["var1_enc"],
             "last_added_predictor": "var1_enc",
             "train_performance": 0.612, "selection_performance": 0.609,
             "model_type": model_type},
            {"predictors": ["var1_enc", "var2_enc"],
             "last_added_predictor": "var2_enc",
             "train_performance": 0.612, "selection_performance": 0.609,
             "model_type": model_type},
            {"predictors": ["var1_enc", "var2_enc", "var3_enc"],
             "last_added_predictor": "var3_enc",
             "train_performance": 0.612, "selection_performance": 0.609,
             "model_type": model_type}
        ])

        pd.testing.assert_frame_equal(actual, expected)

    @pytest.mark.parametrize("model_type, max_predictors, expectation",
                             [("classification", 2, pytest.raises(ValueError)),
                              ("classification", 3, does_not_raise()),
                              ("classification", 5, does_not_raise()),
                              ("classification", 10, does_not_raise()),
                              ("classification", 15, does_not_raise()),
                              ("regression", 2, pytest.raises(ValueError)),
                              ("regression", 3, does_not_raise()),
                              ("regression", 5, does_not_raise()),
                              ("regression", 10, does_not_raise()),
                              ("regression", 15, does_not_raise())
                              ])
    def test_fit(self, mocker, model_type, max_predictors: int, expectation):

        # create list of elements [var1_enc, var2_enc, ..., var10_enc]
        predictors_list = [f"var{i+1}_enc" for i in range(10)]
        # extract sublist [var1_enc, var5_enc, var9_enc]
        forced_predictors_list = predictors_list[::4]

        ordered_output_list = (forced_predictors_list
                               + [pred for pred in predictors_list
                                  if pred not in forced_predictors_list])

        fw_selection = ForwardFeatureSelection(model_type=model_type, max_predictors=max_predictors)

        def mock_train_model(self, train_data, target_column_name, predictors):
            return mock_model(predictors, model_type=model_type)

        def mock_forward_selection(self, train_data, target_column_name,
                                   predictors, forced_predictors):
            n_models = min(max_predictors, len(predictors) + len(forced_predictors))

            return [mock_model(ordered_output_list[:i+1], model_type=model_type)
                    for i in range(n_models)]

        mocker.patch("cobra.model_building.ForwardFeatureSelection._train_model",
                     mock_train_model)

        mocker.patch("cobra.model_building.ForwardFeatureSelection._forward_selection",
                     mock_forward_selection)

        with expectation:
            fw_selection.fit(pd.DataFrame(), "target",
                             predictors=predictors_list,
                             forced_predictors=forced_predictors_list,
                             excluded_predictors=[])

            # for each fitted model, check number of predictors
            actual = [model.predictors
                      for model in fw_selection._fitted_models]

            expected = [ordered_output_list[:i+1]
                        for i in range(min(max_predictors,
                                           len(predictors_list)))]

            if max_predictors == len(forced_predictors_list):
                expected = [forced_predictors_list]

            assert actual == expected

    @pytest.mark.parametrize("model_type, max_predictors", [("classification", 5),
                                                            ("classification", 10),
                                                            ("classification", 15),
                                                            ("regression", 5),
                                                            ("regression", 10),
                                                            ("regression", 15)
                                                            ])
    def test_forward_selection(self, mocker, model_type, max_predictors: int):

        # create list of elements [var1_enc, var2_c, ..., var10_enc]
        predictors_list = [f"var{i+1}_enc" for i in range(10)]

        # extract sublist [var1_enc, var5_enc, var9_enc]:
        forced_predictors = predictors_list[::4]
        # remove these from predictors list to have clean version
        predictors = [pred for pred in predictors_list
                      if pred not in forced_predictors]

        ordered_output_list = forced_predictors + predictors

        def mock_find_next_best_model(self, train_data, target_column_name,
                                      candidate_predictors,
                                      current_predictors):
            return mock_model(current_predictors + candidate_predictors[0:1], model_type=model_type)

        mocker.patch(("cobra.model_building.ForwardFeatureSelection."
                      "_find_next_best_model"), mock_find_next_best_model)

        fw_selection = ForwardFeatureSelection(model_type=model_type, max_predictors=max_predictors)

        fitted_models = (fw_selection.
                         _forward_selection(pd.DataFrame(), "target",
                                            predictors,
                                            forced_predictors))

        actual = [sorted(model.predictors) for model in fitted_models]

        expected = [sorted(ordered_output_list[:i+1])
                    for i in range(min(max_predictors,
                                       len(predictors_list)))]

        assert actual == expected
