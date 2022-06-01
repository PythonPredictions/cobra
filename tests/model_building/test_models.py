import numpy as np
import pandas as pd

from cobra.model_building.models import LogisticRegressionModel, \
    LinearRegressionModel, Model


def mock_data():
    return pd.DataFrame({"var1_enc": [0.42] * 10,
                         "var2_enc": [0.94] * 10,
                         "var3_enc": [0.87] * 10})

mock_score_model_output = np.array(
    [0.5, 0.8, 0.2, 0.9, 0.1, 0.7, 0.3, 0.6, 0.4, 0.5]
)

def mock_score_model(self, data):
    return mock_score_model_output

default_metric_output = 0.17
def mock_evaluate_with_default_metric(self, y_true, y_score):
    return default_metric_output

def mock_prepare_args_for_custom_evaluation_metric(self, y_true, y_score,
                                                   metric):
    return {
        "y_true": y_true,
        "y_score": y_score
    }

class TestModel:
    def test_evaluate_returns_precalculated_performance_from_cache(self,
                                                                   mocker):
        """Test whether evaluate() returns a performance, as calculated
        earlier on the same dataset split, from its internal cache."""
        expected = 0.79

        model = Model()
        model._performance_per_split["train"] = expected  # setting the cache

        # passing empty dataframes as input, instead of those that would
        # enable exact re-calculation of the performance, so we know for sure
        # that the cache has provided the answer.
        actual = model.evaluate(pd.DataFrame(),
                                pd.Series(dtype="float64"),
                                split="train")

        assert actual == expected

    def test_evaluate_with_default_metric(self, mocker):
        X = mock_data()
        y = pd.Series([1] * 5 + [0] * 5)

        (mocker
         .patch("cobra.model_building.models.Model.score_model",
                mock_score_model))

        (mocker
         .patch("cobra.model_building.models"
                ".Model._evaluate_with_default_metric",
                mock_evaluate_with_default_metric))

        model = Model()
        actual = model.evaluate(X, y) # metric=None implied
        assert actual == default_metric_output

    def test_evaluate_with_custom_metric(self, mocker):
        X = mock_data()
        y = pd.Series([1] * 5 + [0] * 5)

        (mocker
         .patch("cobra.model_building.models.Model.score_model",
                mock_score_model))

        # we won't test all combinations of args & kwargs that can occur when
        # calling the custom metric function (basic args y_true and y_score +
        # no args and no kwargs / same + args but no kwargs / etc.), just the
        # combination of basic args, args and kwargs.
        # The child classes will additionally test with some common metrics
        # used with these child models.
        (mocker
         .patch("cobra.model_building.models"
                ".Model._prepare_args_for_custom_evaluation_metric",
                mock_prepare_args_for_custom_evaluation_metric))

        expected_arg1 = 14
        expected_kwarg1 = 17
        expected_custom_metric_output = 0.56
        def some_custom_metric(y_true, y_score,
                               arg1=None,
                               *, kwarg1=None):
            if not np.array_equal(y_true, y):
                raise ValueError("evaluate() did not succeed in correctly "
                                 "passing y_true to the custom metric.")
            if not np.array_equal(y_score, mock_score_model_output):
                raise ValueError("evaluate() did not succeed in correctly "
                                 "passing y_score to the custom metric.")
            if arg1 != expected_arg1:
                raise ValueError("evaluate() did not succeed in correctly "
                                 "passing additional arguments to the custom "
                                 "metric.")
            if kwarg1 != expected_kwarg1:
                raise ValueError("evaluate() did not succeed in correctly "
                                 "passing additional keyword arguments to "
                                 "the custom metric.")
            return expected_custom_metric_output

        model = Model()
        actual = model.evaluate(X, y,
                                metric=some_custom_metric,
                                metric_args={
                                    "arg1": expected_arg1
                                },
                                metric_kwargs={
                                    "kwarg1": expected_kwarg1
                                })
        assert actual == expected_custom_metric_output

    def test_evaluate_caches_performance(self, mocker):
        """Test whether the evaluate() function caches the calculated model
        performance for a certain dataset split."""
        X = mock_data()
        y = pd.Series([1] * 5 + [0] * 5)

        (mocker
         .patch("cobra.model_building.models.Model.score_model",
                mock_score_model))

        (mocker
         .patch("cobra.model_building.models"
                ".Model._evaluate_with_default_metric",
                mock_evaluate_with_default_metric))

        model = Model()
        actual = model.evaluate(X, y,
                                split="train")

        assert model._performance_per_split["train"] == default_metric_output

    def test_compute_variable_importance(self, mocker):

        def mock_pearsonr(ypred, ytrue):
            return [ypred.unique()[0]]

        (mocker
         .patch("cobra.model_building.LogisticRegressionModel.score_model",
                mock_score_model_classification))

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


mock_score_model_classification_output = mock_score_model_output

def mock_score_model_classification(self, data):
    return mock_score_model_classification_output


class TestLogisticRegressionModel:

    # The following are more like integration tests, which verify
    # Model.evaluate() with a few examples of metrics that cover the most use
    # cases that Cobra developers will use when developing a regression model:

    def test_evaluate_no_metric_specified(self, mocker):
        X = mock_data()
        y = pd.Series([1] * 5 + [0] * 5)

        (mocker
         .patch("cobra.model_building.LogisticRegressionModel.score_model",
                mock_score_model_classification))

        def mock_roc_auc_score(y_true, y_score):
            # mocking sklearn.metrics.roc_auc_score, as instantiated in
            # models.py.
            if not np.array_equal(y_true, y):
                raise ValueError("LogisticRegressionModel.evaluate() did not "
                                 "succeed in passing the correct y_true "
                                 "argument.")
            if not np.array_equal(y_score,
                                  mock_score_model_classification_output):
                raise ValueError("LogisticRegressionModel.evaluate() did not "
                                 "succeed in passing the correct y_score "
                                 "argument.")

            return 0.79

        (mocker
         .patch("cobra.model_building.models.roc_auc_score",
                mock_roc_auc_score))

        model = LogisticRegressionModel()
        actual = model.evaluate(X, y)  # implied: metric=None (default value).

        assert actual == 0.79

    def test_evaluate_metric_specified_requiring_y_score(self, mocker):
        X = mock_data()
        y = pd.Series([1] * 5 + [0] * 5)

        (mocker
         .patch("cobra.model_building.LogisticRegressionModel.score_model",
                mock_score_model_classification))

        def top_k_accuracy_score(y_true, y_score,
                                 *, k=2, normalize=True,
                                 sample_weight=None, labels=None):
            # mimicking sklearn.metrics.top_k_accuracy_score.
            if not np.array_equal(y_true, y):
                raise ValueError("LogisticRegressionModel.evaluate() did not "
                                 "succeed in passing the correct y_true "
                                 "argument.")
            if not np.array_equal(y_score,
                                  mock_score_model_classification_output):
                raise ValueError("LogisticRegressionModel.evaluate() did not "
                                 "succeed in passing the correct y_score "
                                 "argument.")

            return 0.14

        model = LogisticRegressionModel()
        actual = model.evaluate(X, y,
                                metric=top_k_accuracy_score)

        assert actual == 0.14

    def test_evaluate_metric_specified_requiring_y_prob(self, mocker):
        X = mock_data()
        y = pd.Series([1] * 5 + [0] * 5)

        (mocker
         .patch("cobra.model_building.LogisticRegressionModel.score_model",
                mock_score_model_classification))

        def brier_score_loss(y_true, y_prob,
                             *, sample_weight=None, pos_label=None):
            # mimicking sklearn.metrics.brier_score_loss
            if not np.array_equal(y_true, y):
                raise ValueError("LogisticRegressionModel.evaluate() did not "
                                 "succeed in passing the correct y_true "
                                 "argument.")
            if not np.array_equal(y_prob,
                                  mock_score_model_classification_output):
                raise ValueError("LogisticRegressionModel.evaluate() did not "
                                 "succeed in passing the correct y_prob "
                                 "argument.")

            return 0.14

        model = LogisticRegressionModel()
        actual = model.evaluate(X, y,
                                metric=brier_score_loss)

        assert actual == 0.14

    def test_evaluate_metric_specified_requiring_y_pred(self, mocker):
        X = mock_data()
        y = pd.Series([1] * 5 + [0] * 5)

        (mocker
         .patch("cobra.model_building.LogisticRegressionModel.score_model",
                mock_score_model_classification))

        def f1_score(y_true, y_pred,
                     *, labels=None, pos_label=1, average='binary',
                     sample_weight=None, zero_division='warn'):
            if not np.array_equal(y_true, y):
                raise ValueError("LogisticRegressionModel.evaluate() did not "
                                 "succeed in passing the correct y_true "
                                 "argument.")
            if not np.array_equal(y_pred,
                                  np.zeros(
                                      (len(mock_score_model_classification_output),)
                                  )):
                raise ValueError("LogisticRegressionModel.evaluate() did not "
                                 "succeed in passing the correct y_pred "
                                 "argument.")

            return 0.14

        # We don't mock roc_curve, mocking the optimal_cutoff (see below) is
        # enough to guarantee that y_pred will have a pre-determined value
        # that we can test for.

        def mock_compute_optimal_cutoff(fpr: np.ndarray, tpr: np.ndarray,
                                        thresholds: np.ndarray) -> float:
            # Let's return a threshold so high, that all scores will end up
            # below it, which will result in a y_pred being equal to np.zeros().
            return float("inf")

        (mocker
         .patch("cobra.evaluation.evaluator.ClassificationEvaluator"
                "._compute_optimal_cutoff",
                mock_compute_optimal_cutoff))

        model = LogisticRegressionModel()
        actual = model.evaluate(X, y,
                                metric=f1_score)

        assert actual == 0.14

    def test_evaluate_metric_specified_with_additional_metric_args(self,
                                                                   mocker):
        X = mock_data()
        y = pd.Series([1] * 5 + [0] * 5)

        (mocker
         .patch("cobra.model_building.LogisticRegressionModel.score_model",
                mock_score_model_classification))

        def compute_lift(y_true,
                         y_score,
                         lift_at=0.05):
            # Mimicking ClassificationEvaluator._compute_lift()
            if not np.array_equal(y_true, y):
                raise ValueError("LogisticRegressionModel.evaluate() did not "
                                 "succeed in passing the correct y_true "
                                 "argument.")
            if not np.array_equal(y_score,
                                  mock_score_model_classification_output):
                raise ValueError("LogisticRegressionModel.evaluate() did not "
                                 "succeed in passing the correct y_score "
                                 "argument.")
            if lift_at != 0.22:
                raise ValueError("LogisticRegressionModel.evaluate() did not "
                                 "succeed in passing the lift_at argument to "
                                 "the metric function.")
            return 0.14

        model = LogisticRegressionModel()
        actual = model.evaluate(X, y,
                                metric=compute_lift,
                                metric_args={
                                    "lift_at": 0.22
                                })

        assert actual == 0.14

    def test_evaluate_metric_specified_with_additional_metric_kwargs(self,
                                                                     mocker):
        X = mock_data()
        y = pd.Series([1] * 5 + [0] * 5)

        (mocker
         .patch("cobra.model_building.LogisticRegressionModel.score_model",
                mock_score_model_classification))

        def top_k_accuracy_score(y_true, y_score,
                                 *, k=2, normalize=True,
                                 sample_weight=None, labels=None):
            # mimicking sklearn.metrics.top_k_accuracy_score.
            if not np.array_equal(y_true, y):
                raise ValueError("LogisticRegressionModel.evaluate() did not "
                                 "succeed in passing the correct y_true "
                                 "argument.")
            if not np.array_equal(y_score,
                                  mock_score_model_classification_output):
                raise ValueError("LogisticRegressionModel.evaluate() did not "
                                 "succeed in passing the correct y_score "
                                 "argument.")
            if k != 100:
                raise ValueError("LogisticRegressionModel.evaluate() did not "
                                 "succeed in passing the kwarg k to the "
                                 "metric function.")

            return 0.14

        model = LogisticRegressionModel()
        actual = model.evaluate(X, y,
                                metric=top_k_accuracy_score,
                                metric_kwargs={
                                    "k": 100
                                })

        assert actual == 0.14

    def test_serialize(self):

        model = LogisticRegressionModel()
        actual = model.serialize()

        expected = {
            "predictors": [],
            "_performance_per_split": {},
            "meta": "logistic-regression",
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
            "_performance_per_split": {},
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
        assert logit.intercept_.all() == (np.array(model_dict["intercept_"]).all())
        assert logit.coef_.all() == np.array(model_dict["coef_"]).all()

    def test_deserialize_backwards_compat_for_eval_metrics_by_split(self):

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
        assert logit.intercept_.all() == (np.array(model_dict["intercept_"]).all())
        assert logit.coef_.all() == np.array(model_dict["coef_"]).all()


mock_score_model_regression_output = np.array([0.7, 0.2, 0.2, 0.9, 0.7, 0.3, 0.1, 0.4, 0.8, 0.5])*15

def mock_score_model_regression(self, data):
    return mock_score_model_regression_output


class TestLinearRegressionModel:

    # The following are more like integration tests, which verify
    # Model.evaluate() with a few examples of metrics that cover the most use
    # cases that Cobra developers will use when developing a regression model:

    def test_evaluate_no_metric_specified(self, mocker):
        X = mock_data()
        y = pd.Series([1] * 5 + [0] * 5)

        (mocker
         .patch("cobra.model_building.models.LinearRegressionModel.score_model",
                mock_score_model_regression))

        def mock_evaluate_with_default_metric(self, y_true, y_score):
            if not np.array_equal(y_true, y):
                raise ValueError("LinearRegressionModel.evaluate() did not "
                                 "succeed in passing the correct y_true "
                                 "argument.")
            if not np.array_equal(y_score,
                                  mock_score_model_regression_output):
                raise ValueError("LinearRegressionModel.evaluate() did not "
                                 "succeed in passing the correct y_score "
                                 "argument.")

            return 0.79

        (mocker
         .patch("cobra.model_building.models.LinearRegressionModel."
                "_evaluate_with_default_metric",
                mock_evaluate_with_default_metric))

        model = LinearRegressionModel()
        actual = model.evaluate(X, y)  # implied: metric=None (default value).

        assert actual == 0.79

    # no integration tests for specific custom metrics for this model,
    # the tests for LogisticRegressionModel were already elaborate to cover
    # all pathways in Model.evaluate().

    def test_serialize(self):

        model = LinearRegressionModel()
        actual = model.serialize()

        expected = {
            "predictors": [],
            "_performance_per_split": {},
            "meta": "linear-regression",
            "params": {
                "copy_X": True,
                "fit_intercept": True,
                "n_jobs": None,
                "normalize": "deprecated",
                "positive": False
            }
        }

        assert actual == expected

    def test_deserialize(self):

        model = LinearRegressionModel()

        model_dict = {
            "meta": "linear-regression",
            "predictors": [],
            "_performance_per_split": {},
            "params": {
                "copy_X": True,
                "fit_intercept": True,
                "n_jobs": None,
                "normalize": "deprecated",
                "positive": False
            },
            "coef_": [[0.5, 0.75]],
            "intercept_": [-3]
        }

        model.deserialize(model_dict)

        linear = model.linear
        assert linear.get_params() == model_dict["params"]
        assert linear.intercept_.all() == (np.array(model_dict["intercept_"]).all())
        assert linear.coef_.all() == np.array(model_dict["coef_"]).all()

    def test_deserialize_backwards_compat_for_eval_metrics_by_split(self):

        model = LinearRegressionModel()

        model_dict = {
            "meta": "linear-regression",
            "predictors": [],
            "_eval_metrics_by_split": {},
            "params": {
                "copy_X": True,
                "fit_intercept": True,
                "n_jobs": None,
                "normalize": "deprecated",
                "positive": False
            },
            "coef_": [[0.5, 0.75]],
            "intercept_": [-3]
        }

        model.deserialize(model_dict)

        linear = model.linear
        assert linear.get_params() == model_dict["params"]
        assert linear.intercept_.all() == (np.array(model_dict["intercept_"]).all())
        assert linear.coef_.all() == np.array(model_dict["coef_"]).all()
