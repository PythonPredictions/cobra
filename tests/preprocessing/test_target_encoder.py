
import pytest
import pandas as pd
from sklearn.exceptions import NotFittedError

from cobra.preprocessing.target_encoder import TargetEncoder

class TestTargetEncoder:

    def test_target_encoder_constructor_weight_value_error(self):
        with pytest.raises(ValueError):
            TargetEncoder(weight=-1)

    def test_target_encoder_constructor_imputation_value_error(self):
        with pytest.raises(ValueError):
            TargetEncoder(imputation_strategy="median")

    # Tests for attributes_attributes_to_dict and set_attributes_from_dict
    def test_target_encoder_attributes_to_dict(self):
        encoder = TargetEncoder()

        mapping_data = pd.Series(data=[0.333333, 0.50000, 0.666667],
                                 index=["negative", "neutral", "positive"])
        mapping_data.index.name = "variable"

        encoder._mapping["variable"] = mapping_data

        encoder._global_mean = 0.5

        actual = encoder.attributes_to_dict()

        expected = {"weight": 0.0,
                    "imputation_strategy": "mean",
                    "_global_mean": 0.5,
                    "_mapping": {"variable": {
                        "negative": 0.333333,
                        "neutral": 0.50000,
                        "positive": 0.666667
                    }}}

        assert actual == expected

    @pytest.mark.parametrize("attribute",
                             ["weight", "mapping"],
                             ids=["test_weight", "test_mapping"])
    def test_target_encoder_set_attributes_from_dict_unfitted(self, attribute):
        encoder = TargetEncoder()

        data = {"weight": 1.0}
        encoder.set_attributes_from_dict(data)

        if attribute == "weight":
            actual = encoder.weight
            expected = 1.0

            assert expected == actual
        elif attribute == "mapping":
            actual = encoder._mapping
            expected = {}

            assert expected == actual

    def test_target_encoder_set_attributes_from_dict(self):
        encoder = TargetEncoder()

        data = {"weight": 0.0,
                "_global_mean": 0.5,
                "_mapping": {"variable": {
                    "negative": 0.333333,
                    "neutral": 0.50000,
                    "positive": 0.666667
                }}}

        encoder.set_attributes_from_dict(data)

        expected = pd.Series(data=[0.333333, 0.50000, 0.666667],
                             index=["negative", "neutral", "positive"])
        expected.index.name = "variable"

        actual = encoder._mapping["variable"]

        pd.testing.assert_series_equal(actual, expected)

    # Tests for _fit_column:
    def test_target_encoder_fit_column_binary_classification(self):
        df = pd.DataFrame({'variable': ['positive', 'positive', 'negative',
                                        'neutral', 'negative', 'positive',
                                        'negative', 'neutral', 'neutral',
                                        'neutral'],
                           'target': [1, 1, 0, 0, 1, 0, 0, 0, 1, 1]})

        encoder = TargetEncoder()
        encoder._global_mean = 0.5
        actual = encoder._fit_column(X=df.variable, y=df.target)

        expected = pd.Series(data=[0.333333, 0.50000, 0.666667],
                             index=["negative", "neutral", "positive"])
        expected.index.name = "variable"

        pd.testing.assert_series_equal(actual, expected)

    def test_target_encoder_fit_column_linear_regression(self):
        df = pd.DataFrame({'variable': ['positive', 'positive', 'negative',
                                        'neutral', 'negative', 'positive',
                                        'negative', 'neutral', 'neutral',
                                        'neutral', 'positive'],
                           'target': [5, 4, -5, 0, -4, 5, -5, 0, 1, 0, 4]})

        encoder = TargetEncoder()
        encoder._global_mean = 0.454545
        actual = encoder._fit_column(X=df.variable, y=df.target)

        expected = pd.Series(data=[-4.666667, 0.250000, 4.500000],
                             index=["negative", "neutral", "positive"])
        expected.index.name = "variable"

        pd.testing.assert_series_equal(actual, expected)

    def test_target_encoder_fit_column_global_mean_binary_classification(self):
        df = pd.DataFrame({'variable': ['positive', 'positive', 'negative',
                                        'neutral', 'negative', 'positive',
                                        'negative', 'neutral', 'neutral',
                                        'neutral'],
                           'target': [1, 1, 0, 0, 1, 0, 0, 0, 1, 1]})

        encoder = TargetEncoder(weight=1)
        encoder._global_mean = df.target.sum() / df.target.count()  # is 0.5

        actual = encoder._fit_column(X=df.variable, y=df.target)

        expected = pd.Series(data=[0.375, 0.500, 0.625],
                             index=["negative", "neutral", "positive"])
        expected.index.name = "variable"

        pd.testing.assert_series_equal(actual, expected)

    def test_target_encoder_fit_column_global_mean_linear_regression(self):
        df = pd.DataFrame({'variable': ['positive', 'positive', 'negative',
                                        'neutral', 'negative', 'positive',
                                        'negative', 'neutral', 'neutral',
                                        'neutral', 'positive'],
                           'target': [5, 4, -5, 0, -4, 5, -5, 0, 1, 0, 4]})

        encoder = TargetEncoder(weight=1)
        encoder._global_mean = 0.454545

        actual = encoder._fit_column(X=df.variable, y=df.target)

        # expected new value:
        # [count of the value * its mean encoding + weight (= 1) * global mean]
        # / [count of the value + weight (=1)].
        expected = pd.Series(data=[(3 * -4.666667 + 1 * 0.454545) / (3 + 1),
                                   (4 * 0.250000 + 1 * 0.454545) / (4 + 1),
                                   (4 * 4.500000 + 1 * 0.454545) / (4 + 1)],
                             index=["negative", "neutral", "positive"])
        expected.index.name = "variable"

        pd.testing.assert_series_equal(actual, expected)

    # Tests for fit method
    def test_target_encoder_fit_binary_classification(self):
        # test_target_encoder_fit_column_linear_regression() tested on one
        # column input as a numpy series; this test runs on a dataframe input.
        df = pd.DataFrame({'variable': ['positive', 'positive', 'negative',
                                        'neutral', 'negative', 'positive',
                                        'negative', 'neutral', 'neutral',
                                        'neutral'],
                           'target': [1, 1, 0, 0, 1, 0, 0, 0, 1, 1]})

        encoder = TargetEncoder()
        encoder.fit(data=df, column_names=["variable"], target_column="target")

        expected = pd.Series(data=[0.333333, 0.50000, 0.666667],
                             index=["negative", "neutral", "positive"])
        expected.index.name = "variable"
        actual = encoder._mapping["variable"]

        pd.testing.assert_series_equal(actual, expected)

    def test_target_encoder_fit_linear_regression(self):
        # test_target_encoder_fit_column_linear_regression() tested on one
        # column input as a numpy series; this test runs on a dataframe input.
        df = pd.DataFrame({'variable': ['positive', 'positive', 'negative',
                                        'neutral', 'negative', 'positive',
                                        'negative', 'neutral', 'neutral',
                                        'neutral', 'positive'],
                           'target': [5, 4, -5, 0, -4, 5, -5, 0, 1, 0, 4]})

        encoder = TargetEncoder()
        encoder.fit(data=df, column_names=["variable"], target_column="target")

        expected = pd.Series(data=[-4.666667, 0.250000, 4.500000],
                             index=["negative", "neutral", "positive"])
        expected.index.name = "variable"
        actual = encoder._mapping["variable"]

        pd.testing.assert_series_equal(actual, expected)

    # Tests for transform method
    def test_target_encoder_transform_when_not_fitted(self):
        df = pd.DataFrame({'variable': ['positive', 'positive', 'negative',
                                        'neutral', 'negative', 'positive',
                                        'negative', 'neutral', 'neutral',
                                        'neutral'],
                           'target': [1, 1, 0, 0, 1, 0, 0, 0, 1, 1]})

        # inputs of TargetEncoder will be of dtype category
        df["variable"] = df["variable"].astype("category")

        encoder = TargetEncoder()
        with pytest.raises(NotFittedError):
            encoder.transform(data=df, column_names=["variable"])

    def test_target_encoder_transform_binary_classification(self):
        df = pd.DataFrame({'variable': ['positive', 'positive', 'negative',
                                        'neutral', 'negative', 'positive',
                                        'negative', 'neutral', 'neutral',
                                        'neutral'],
                           'target': [1, 1, 0, 0, 1, 0, 0, 0, 1, 1]})

        # inputs of TargetEncoder will be of dtype category
        df["variable"] = df["variable"].astype("category")

        expected = df.copy()
        expected["variable_enc"] = [0.666667, 0.666667, 0.333333, 0.50000,
                                    0.333333, 0.666667, 0.333333, 0.50000,
                                    0.50000, 0.50000]

        encoder = TargetEncoder()
        encoder.fit(data=df, column_names=["variable"], target_column="target")
        actual = encoder.transform(data=df, column_names=["variable"])

        pd.testing.assert_frame_equal(actual, expected)

    def test_target_encoder_transform_linear_regression(self):
        df = pd.DataFrame({'variable': ['positive', 'positive', 'negative',
                                        'neutral', 'negative', 'positive',
                                        'negative', 'neutral', 'neutral',
                                        'neutral', 'positive'],
                           'target': [5, 4, -5, 0, -4, 5, -5, 0, 1, 0, 4]})

        # inputs of TargetEncoder will be of dtype category
        df["variable"] = df["variable"].astype("category")

        expected = df.copy()
        expected["variable_enc"] = [4.500000, 4.500000, -4.666667, 0.250000,
                                    -4.666667, 4.500000, -4.666667, 0.250000,
                                    0.250000, 0.250000, 4.500000]

        encoder = TargetEncoder()
        encoder.fit(data=df, column_names=["variable"], target_column="target")
        actual = encoder.transform(data=df, column_names=["variable"])

        pd.testing.assert_frame_equal(actual, expected)

    def test_target_encoder_transform_new_category_binary_classification(self):
        df = pd.DataFrame({'variable': ['positive', 'positive', 'negative',
                                        'neutral', 'negative', 'positive',
                                        'negative', 'neutral', 'neutral',
                                        'neutral'],
                           'target': [1, 1, 0, 0, 1, 0, 0, 0, 1, 1]})

        df_appended = pd.concat([df, pd.DataFrame({"variable": "new", "target": 1}, index=[len(df)])], ignore_index=True)

        # inputs of TargetEncoder will be of dtype category
        df["variable"] = df["variable"].astype("category")
        df_appended["variable"] = df_appended["variable"].astype("category")

        expected = df_appended.copy()
        expected["variable_enc"] = [0.666667, 0.666667, 0.333333, 0.50000,
                                    0.333333, 0.666667, 0.333333, 0.50000,
                                    0.50000, 0.50000, 0.333333]

        encoder = TargetEncoder(imputation_strategy="min")
        encoder.fit(data=df, column_names=["variable"], target_column="target")
        actual = encoder.transform(data=df_appended, column_names=["variable"])

        pd.testing.assert_frame_equal(actual, expected)

    def test_target_encoder_transform_new_category_linear_regression(self):
        df = pd.DataFrame({'variable': ['positive', 'positive', 'negative',
                                        'neutral', 'negative', 'positive',
                                        'negative', 'neutral', 'neutral',
                                        'neutral', 'positive'],
                           'target': [5, 4, -5, 0, -4, 5, -5, 0, 1, 0, 4]})

        df_appended = pd.concat([df, pd.DataFrame({"variable": "new", "target": 10}, index=[len(df)])], ignore_index=True)

        # inputs of TargetEncoder will be of dtype category
        df["variable"] = df["variable"].astype("category")
        df_appended["variable"] = df_appended["variable"].astype("category")

        expected = df_appended.copy()
        expected["variable_enc"] = [4.500000, 4.500000, -4.666667, 0.250000,
                                    -4.666667, 4.500000, -4.666667, 0.250000,
                                    0.250000, 0.250000, 4.500000,
                                    -4.666667] # min imputation for new value

        encoder = TargetEncoder(imputation_strategy="min")
        encoder.fit(data=df, column_names=["variable"], target_column="target")
        actual = encoder.transform(data=df_appended, column_names=["variable"])

        pd.testing.assert_frame_equal(actual, expected)

    # Tests for _clean_column_name:
    def test_target_encoder_clean_column_name_binned_column(self):
        column_name = "test_column_bin"
        expected = "test_column_enc"

        encoder = TargetEncoder()
        actual = encoder._clean_column_name(column_name)

        assert actual == expected

    def test_target_encoder_clean_column_name_processed_column(self):
        column_name = "test_column_processed"
        expected = "test_column_enc"

        encoder = TargetEncoder()
        actual = encoder._clean_column_name(column_name)

        assert actual == expected

    def test_target_encoder_clean_column_name_cleaned_column(self):
        column_name = "test_column_cleaned"
        expected = "test_column_enc"

        encoder = TargetEncoder()
        actual = encoder._clean_column_name(column_name)

        assert actual == expected

    def test_target_encoder_clean_column_other_name(self):
        column_name = "test_column"
        expected = "test_column_enc"

        encoder = TargetEncoder()
        actual = encoder._clean_column_name(column_name)

        assert actual == expected
