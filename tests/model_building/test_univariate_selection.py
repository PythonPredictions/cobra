
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
