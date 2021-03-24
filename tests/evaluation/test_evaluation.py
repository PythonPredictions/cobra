import pytest
import pandas as pd
import numpy as np
from cobra.evaluation import plot_incidence
from cobra.evaluation import Evaluator

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

    def test_plot_incidence(self):
        data = mock_data()
        column_order = ['1st-4th', '5th-6th', '7th-8th']
        with pytest.raises(Exception):
            plot_incidence(data, 'education', column_order)

    def test_lift_curve_n_bins(self):
        n_bins_test = [5, 10, 15, 35]

        y_true, y_pred = mock_preds(50)

        n_bins_out = []
        for n_bins in n_bins_test:
            e = Evaluator(n_bins = n_bins)
            out = e._compute_lift_per_bin(y_true, y_pred, e.n_bins)
            lifts = out[1]
            n_bins_out.append(len(lifts))

        assert n_bins_test == n_bins_out
