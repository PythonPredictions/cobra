
import pytest
import pandas as pd
import numpy as np

from cobra.evaluation import plot_incidence
from cobra.evaluation import ClassificationEvaluator, RegressionEvaluator

def mock_data():
    d = {'variable': ['education', 'education', 'education', 'education'],
         'label': ['1st-4th', '5th-6th', '7th-8th', '9th'],
         'pop_size': [0.002, 0.004, 0.009, 0.019],
         'avg_incidence': [0.23, 0.23, 0.23, 0.23],
         'incidence': [0.047, 0.0434, 0.054, 0.069]}
    return pd.DataFrame(d)

def mock_preds(n, seed = 505):
    np.random.seed(seed)

    y_true = np.random.uniform(size=n)
    y_pred = np.random.uniform(size=n)

    return y_true, y_pred

class TestEvaluation:

    def test_plot_incidence_with_unsupported_model_type(self):
        with pytest.raises(ValueError):
            plot_incidence(pig_tables=None,
                           variable="",
                           model_type="anomaly_detection")

    def test_plot_incidence_with_different_column_orders(self):
        data = mock_data()
        with pytest.raises(ValueError):
            plot_incidence(pig_tables=data,
                           variable='education',
                           model_type="classification",
                           # different bins than in the data variable:
                           column_order=['1st-4th', '5th-6th', '7th-8th'])

    # Stubs for later (requires exposing df_plot and testing matplotlib's
    # plot object fix and ax internals):
    """
    def test_plot_incidence_without_column_order(self):
        data = mock_data()
        plot_incidence(pig_tables=data, 
                       variable='education',
                       model_type="classification",
                       column_order=None)

    def test_plot_incidence_with_column_order(self):
        data = mock_data()
        plot_incidence(pig_tables=data,
                       variable='education',
                       model_type="classification",
                       column_order=['1st-4th', '5th-6th', '7th-8th', '9th'])
        
    def test_plot_incidence_visual_result_for_classification(self):
        data = mock_data()
        plot_incidence(pig_tables=data,
                       variable='education',
                       model_type="classification",
                       column_order=['1st-4th', '5th-6th', '7th-8th', '9th'])
    
    def test_plot_incidence_visual_result_for_regression(self):
        data = mock_data()  # change into regression target though.
        plot_incidence(pig_tables=data,
                       variable='education',
                       model_type="regression",
                       column_order=['1st-4th', '5th-6th', '7th-8th', '9th'])
        
    def test_plot_predictions_regression(self):
        y_true, y_pred = mock_preds(50, seed=123)

        evaluator = RegressionEvaluator()
        evaluator.fit(y_true, y_pred)
        evaluator.plot_predictions()
        
    def test_plot_qq(self):
        y_true, y_pred = mock_preds(50, seed=631993)

        evaluator = RegressionEvaluator()
        evaluator.fit(y_true, y_pred)
        evaluator.plot_qq()
    """

    def test_lift_curve_n_bins(self):
        n_bins_test = [5, 10, 15, 35]

        y_true, y_pred = mock_preds(50)

        n_bins_out = []
        for n_bins in n_bins_test:
            e = ClassificationEvaluator(n_bins=n_bins)
            out = ClassificationEvaluator._compute_lift_per_bin(y_true, y_pred, e.n_bins)
            lifts = out[1]
            n_bins_out.append(len(lifts))

        assert n_bins_test == n_bins_out

    def test_fit_classification(self):
        y_true, y_pred = mock_preds(50)
        y_true = (y_true > 0.5).astype(int)  # convert to 0-1 labels

        evaluator = ClassificationEvaluator(n_bins=5)
        evaluator.fit(y_true, y_pred)

        assert (evaluator.y_true == y_true).all()
        assert (evaluator.y_pred == y_pred).all()
        for metric in ["accuracy", "AUC", "precision", "recall",
                       "F1", "matthews_corrcoef", "lift at {}".format(evaluator.lift_at)]:
            assert evaluator.scalar_metrics[metric] is not None
        assert evaluator.roc_curve is not None
        assert evaluator.confusion_matrix is not None
        assert evaluator.lift_curve is not None
        assert evaluator.cumulative_gains is not None

    def test_fit_regression(self):
        y_true, y_pred = mock_preds(50, seed=789)
        y_true, y_pred = y_true*10, y_pred*10  # rescale so it looks more regression-like
        evaluator = RegressionEvaluator()
        evaluator.fit(y_true, y_pred)

        assert (evaluator.y_true == y_true).all()
        assert (evaluator.y_pred == y_pred).all()
        for metric in ["R2", "MAE", "MSE", "RMSE"]:
            assert evaluator.scalar_metrics[metric] is not None
        assert evaluator.qq is not None
