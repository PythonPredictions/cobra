
import pandas as pd
import pytest

from cobra.model_building import univariate_selection


@pytest.fixture
def mock_data():
    return pd.DataFrame({"var1_enc": [0.42] * 10,
                         "var2_enc": [0.94] * 10,
                         "var3_enc": [0.87] * 10})

class TestUnivariateSelection:

    def test_preselection_classification(self, mock_data: pd.DataFrame):
        X = mock_data
        y = pd.DataFrame([1] * 5 + [0] * 5, columns=["target"])

        basetable = pd.concat([y, X], axis=1)
        basetable["split"] = ["train"] * 3 + ["selection"] * 6 + ["train"]

        df_auc = univariate_selection.compute_univariate_preselection(
            target_enc_train_data=basetable[basetable["split"] == "train"],
            target_enc_selection_data=basetable[basetable["split"] == "selection"],
            predictors=X.columns,
            target_column="target",
            model_type="classification",
            preselect_auc_threshold=0.48,
            preselect_overtrain_threshold=0.05)

        assert all(c in df_auc.columns for c in ["AUC train", "AUC selection"])

        preselected_predictors = univariate_selection.get_preselected_predictors(df_auc)
        assert preselected_predictors == ["var1_enc", "var2_enc", "var3_enc"]

    def test_preselection_regression(self, mock_data: pd.DataFrame):
        X = mock_data
        y = pd.DataFrame([6.0, 9.0, 4.2, 5.5, 0.7, 1.9, 8.7, 8.0, 2.0, 7.2], columns=["target"])

        basetable = pd.concat([y, X], axis=1)
        basetable["split"] = ["train"] * 3 + ["selection"] * 6 + ["train"]

        df_rmse = univariate_selection.compute_univariate_preselection(
            target_enc_train_data=basetable[basetable["split"] == "train"],
            target_enc_selection_data=basetable[basetable["split"] == "selection"],
            predictors=X.columns,
            target_column="target",
            model_type="regression",
            preselect_auc_threshold=5,
            preselect_overtrain_threshold=0.05)

        assert all(c in df_rmse.columns for c in ["RMSE train", "RMSE selection"])

        preselected_predictors = univariate_selection.get_preselected_predictors(df_rmse)
        assert preselected_predictors == ["var2_enc", "var3_enc"]

    def test_filter_preselection_error_based(self):
        """Test filtering preselection data for an error-based metric."""
        test_input = pd.DataFrame(
            [
                [0.1, 0.1],
                [0.2, 0.2],
                [0.3, 0.6],
                [0.4, 0.4],
                [0.5, 0.5],
                [0.6, 0.6],
                [0.7, 0.7],
                [0.8, 0.8],
                [0.9, 0.9],
                [1.0, 1.0],
            ],
            columns=["RMSE train", "RMSE selection"]
        )
        result = univariate_selection.filter_preselection_error_based(
            test_input,
            preselect_threshold=0.65,
            preselect_overtrain=0.2,
            scoring_method="RMSE"
        )

        target = pd.DataFrame(
            [
                [0.1, 0.1, True],
                [0.2, 0.2, True],
                [0.4, 0.4, True],
                [0.5, 0.5, True],
                [0.3, 0.6, False],
                [0.6, 0.6, True],
                [0.7, 0.7, False],
                [0.8, 0.8, False],
                [0.9, 0.9, False],
                [1.0, 1.0, False],
            ],
            columns=["RMSE train", "RMSE selection", "preselection"]
        )
        assert target.equals(result)

    def test_filter_preselection_score_based(self):
        """Test filtering preselection data for a score-based metric."""
        test_input = pd.DataFrame(
            [
                [0.1, 0.1],
                [0.2, 0.2],
                [0.3, 0.6],
                [0.4, 0.4],
                [0.5, 0.5],
                [0.6, 0.6],
                [0.7, 0.7],
                [0.8, 0.8],
                [0.9, 0.9],
                [1.0, 0.7],
            ],
            columns=["AUC train", "AUC selection"]
        )
        result = univariate_selection.filter_preselection_score_based(
            test_input,
            preselect_threshold=0.65,
            preselect_overtrain=0.2,
            scoring_method="AUC"
        )

        target = pd.DataFrame(
            [
                [0.9, 0.9, True],
                [0.8, 0.8, True],
                [0.7, 0.7, True],
                [1.0, 0.7, False],
                [0.3, 0.6, False],
                [0.6, 0.6, False],
                [0.5, 0.5, False],
                [0.4, 0.4, False],
                [0.2, 0.2, False],
                [0.1, 0.1, False],
            ],
            columns=["AUC train", "AUC selection", "preselection"]
        )
        assert target.equals(result)
