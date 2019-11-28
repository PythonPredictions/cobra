import pytest
import pandas as pd

from cobra.preprocessing.target_encoder import TargetEncoder


class TestTargetEncoder:

    def test_target_encoder_constructor_value_error(self):
        with pytest.raises(ValueError):
            TargetEncoder(weight=-1)

    # Tests for attributes_attributes_to_dict and set_attributes_from_dict
    def test_target_encoder_attributes_to_dict(self):

        encoder = TargetEncoder()

        mapping_data = pd.Series(data=[0.333333, 0.50000, 0.666667],
                                 index=["negative", "neutral", "positive"])
        mapping_data.index.name = "variable"

        encoder._mapping["variable"] = mapping_data

        actual = encoder.attributes_to_dict()

        expected = {"weight": 0.0,
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

        pd.testing.assert_series_equal(actual, expected,
                                       check_less_precise=5)

    # Tests for _fit_column
    def test_target_encoder_fit_column(self):

        df = pd.DataFrame({'variable': ['positive', 'positive', 'negative',
                                        'neutral', 'negative', 'positive',
                                        'negative', 'neutral', 'neutral',
                                        'neutral'],
                           'target': [1, 1, 0, 0, 1, 0, 0, 0, 1, 1]})

        encoder = TargetEncoder()
        actual = encoder._fit_column(X=df.variable, y=df.target,
                                     global_mean=0.0)

        expected = pd.Series(data=[0.333333, 0.50000, 0.666667],
                             index=["negative", "neutral", "positive"])
        expected.index.name = "variable"

        pd.testing.assert_series_equal(actual, expected,
                                       check_less_precise=5)

    def test_target_encoder_fit_column_global_mean(self):

        df = pd.DataFrame({'variable': ['positive', 'positive', 'negative',
                                        'neutral', 'negative', 'positive',
                                        'negative', 'neutral', 'neutral',
                                        'neutral'],
                           'target': [1, 1, 0, 0, 1, 0, 0, 0, 1, 1]})

        global_mean = df.target.sum() / df.target.count()  # is 0.5

        encoder = TargetEncoder(weight=1)
        actual = encoder._fit_column(X=df.variable, y=df.target,
                                     global_mean=global_mean)

        expected = pd.Series(data=[0.375, 0.500, 0.625],
                             index=["negative", "neutral", "positive"])
        expected.index.name = "variable"

        pd.testing.assert_series_equal(actual, expected,
                                       check_less_precise=3)

    # Tests for fit method
    def test_target_encoder_fit(self):

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

        pd.testing.assert_series_equal(actual, expected,
                                       check_less_precise=5)

    # Tests for transform method
    def test_target_encoder_transform(self):

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

        pd.testing.assert_frame_equal(actual, expected,
                                      check_less_precise=5)

    # Tests for _clean_column_name
    def test_target_encoder_clean_column_name(self):

        column_name = "test_column"
        expected = "test_column_enc"

        encoder = TargetEncoder()
        actual = encoder._clean_column_name(column_name)

        assert actual == expected

    def test_target_encoder_clean_column_name_binned_column(self):

        column_name = "test_column_bin"
        expected = "test_column_enc"

        encoder = TargetEncoder()
        actual = encoder._clean_column_name(column_name)

        assert actual == expected
