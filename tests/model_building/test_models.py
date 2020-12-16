import numpy as np
import pandas as pd

from cobra.model_building.models import LogisticRegressionModel


def mock_data():
    return pd.DataFrame({"var1_enc": [0.42] * 10,
                         "var2_enc": [0.94] * 10,
                         "var3_enc": [0.87] * 10})


def mock_score_model(self, data):
    return np.array([0.5, 0.8, 0.2, 0.9, 0.1, 0.7, 0.3, 0.6, 0.4, 0.5])


class TestLogisticRegressionModel:

    def test_evaluate(self, mocker):

        X = mock_data()
        y = pd.Series([1] * 5 + [0] * 5)

        def mock_roc_auc_score(y_true, y_score):
            return 0.79

        (mocker
         .patch("cobra.model_building.LogisticRegressionModel.score_model",
                mock_score_model))

        (mocker
         .patch("cobra.model_building.models.roc_auc_score",
                mock_roc_auc_score))

        model = LogisticRegressionModel()
        actual = model.evaluate(X, y)

        assert actual == 0.79

    def test_evaluate_cached(self):

        split = "train"
        expected = 0.79

        model = LogisticRegressionModel()
        model._eval_metrics_by_split["train"] = expected

        actual = model.evaluate(pd.DataFrame(), pd.Series(dtype="float64"),
                                split)

        assert actual == expected

    def test_compute_variable_importance(self, mocker):

        def mock_pearsonr(ypred, ytrue):
            return [ypred.unique()[0]]

        (mocker
         .patch("cobra.model_building.LogisticRegressionModel.score_model",
                mock_score_model))

        (mocker
         .patch("cobra.model_building.models.stats.pearsonr",
                mock_pearsonr))

        model = LogisticRegressionModel()
        model.predictors = ["var1_enc", "var2_enc", "var3_enc"]

        data = mock_data()

        actual = model.compute_variable_importance(data)

        expected = pd.DataFrame([
            {"predictor": "var1", "importance": data["var1_enc"].unique()[0]},
            {"predictor": "var2", "importance": data["var2_enc"].unique()[0]},
            {"predictor": "var3", "importance": data["var3_enc"].unique()[0]}
        ]).sort_values(by="importance", ascending=False).reset_index(drop=True)

        pd.testing.assert_frame_equal(actual, expected)

    def test_serialize(self, mocker):

        model = LogisticRegressionModel()
        actual = model.serialize()

        expected = {
            "meta": "logistic-regression",
            "predictors": [],
            "_eval_metrics_by_split": {},
            "params": {
                "C": 1000000000.0,
                "class_weight": None,
                "dual": False,
                "fit_intercept": True,
                "intercept_scaling": 1,
                "l1_ratio": None,
                "max_iter": 100,
                "multi_class": "auto",
                "n_jobs": None,
                "penalty": "l2",
                "random_state": 42,
                "solver": "liblinear",
                "tol": 0.0001,
                "verbose": 0,
                "warm_start": False
            }
        }

        assert actual == expected

    def test_deserialize(self):

        model = LogisticRegressionModel()

        model_dict = {
            "meta": "logistic-regression",
            "predictors": [],
            "_eval_metrics_by_split": {},
            "params": {
                "C": 1000000000.0,
                "class_weight": None,
                "dual": False,
                "fit_intercept": True,
                "intercept_scaling": 1,
                "l1_ratio": None,
                "max_iter": 100,
                "multi_class": "auto",
                "n_jobs": None,
                "penalty": "l2",
                "random_state": 42,
                "solver": "liblinear",
                "tol": 0.0001,
                "verbose": 0,
                "warm_start": False
            },
            "classes_": [0, 1],
            "coef_": [[0.5, 0.75]],
            "intercept_": [-3],
            "n_iter_": [10]
        }

        model.deserialize(model_dict)

        logit = model.logit
        assert logit.get_params() == model_dict["params"]
        assert logit.classes_.all() == np.array(model_dict["classes_"]).all()
        assert logit.n_iter_.all() == np.array(model_dict["n_iter_"]).all()
        assert logit.intercept_.all() == (np.array(model_dict["intercept_"])
                                          .all())
        assert logit.coef_.all() == np.array(model_dict["coef_"]).all()
