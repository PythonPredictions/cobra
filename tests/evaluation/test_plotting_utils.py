from pandas import DataFrame
from cobra.evaluation import (plot_univariate_predictor_quality, 
                              plot_correlation_matrix,
                              plot_performance_curves,
                              plot_variable_importance)

def mock_df_rmse() -> DataFrame:
    return DataFrame(
        {'predictor': {0: 'weight', 1: 'displacement', 2: 'horsepower',
                       3: 'cylinders',  4: 'origin',  5: 'model_year',
                       6: 'name',  7: 'acceleration'},
        'RMSE train': {0: 4.225088318760745, 1: 4.403878881676005,
                       2: 4.3343326307873875, 3: 4.901531871261906,
                       4: 6.6435969708016955, 5: 6.318271823003904,
                       6: 1.4537996193882199, 7: 6.631180878197439},
        'RMSE selection': {0: 4.006855931973032, 1: 4.146696570151399,
                           2: 4.321365764687869, 3: 4.466259266291863,
                           4: 5.833138420191894, 5: 5.979795941821068,
                           6: 6.99641113758452, 7: 7.449190759856361},
        'preselection': {0: True, 1: True, 2: True, 3: True, 4: True,
                         5: True, 6: True, 7: True}}
        )


def mock_df_corr() -> DataFrame:
    return DataFrame({
        'cylinders': {'cylinders': 1.0, 'weight': 0.8767772796304492, 'horsepower': 0.8124872187173973},
        'weight': {'cylinders': 0.8767772796304492, 'weight': 1.0, 'horsepower': 0.8786843186591881},
        'horsepower': {'cylinders': 0.8124872187173973, 'weight': 0.8786843186591881, 'horsepower': 1.0}})

def mock_performances() -> DataFrame:
    return DataFrame({
        'predictors': {0: ['weight_enc'], 1: ['weight_enc', 'horsepower_enc'], 2: ['horsepower_enc', 'weight_enc', 'cylinders_enc']},
        'last_added_predictor': {0: 'weight_enc', 1: 'horsepower_enc', 2: 'cylinders_enc'},
        'train_performance': {0: 4.225088318760745, 1: 3.92118718828259, 2: 3.8929681840552495},
        'selection_performance': {0: 4.006855931973032, 1: 3.630079770314085, 2: 3.531305702221386},
        'validation_performance': {0: 4.348180862267973, 1: 4.089638309577036, 2: 3.9989641017455995},
        'model_type': {0: 'regression', 1: 'regression', 2: 'regression'}
        })

def mock_variable_importance() -> DataFrame:
    return DataFrame({
        'predictor': {0: 'weight', 1: 'horsepower', 2: 'model_year', 3: 'origin'},
        'importance': {0: 0.8921354566046729, 1: 0.864633073581914, 2: 0.694399044392948, 3: 0.6442243718390968}
        })

def test_plot_univariate_predictor_quality():
    plot_univariate_predictor_quality(mock_df_rmse())

def test_plot_correlation_matrix():
    plot_correlation_matrix(mock_df_corr())

def test_plot_performance_curves():
    plot_performance_curves(mock_performances())

def test_plot_variable_importance():
    plot_variable_importance(mock_variable_importance())
