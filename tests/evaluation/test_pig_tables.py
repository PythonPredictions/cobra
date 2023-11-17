import pytest
import pandas as pd

from cobra.evaluation import plot_incidence

def mock_data():
    d = {'variable': ['education', 'education', 'education', 'education'],
         'label': ['1st-4th', '5th-6th', '7th-8th', '9th'],
         'pop_size': [0.002, 0.004, 0.009, 0.019],
         'global_avg_target': [0.23, 0.23, 0.23, 0.23],
         'avg_target': [0.047, 0.0434, 0.054, 0.069]}
    return pd.DataFrame(d)


def test_plot_incidence():
    plot_incidence(pig_tables=mock_data(),
                   variable="education",
                   model_type="regression",)

