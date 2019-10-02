import pytest
import pandas as pd

from cobra.preprocessing.target_encoder import TargetEncoder


class TestTargetEncoder:

    def test_target_encoder_constructor_value_error(self):
        with pytest.raises(ValueError):
            TargetEncoder(weight=-1)

    # Tests for _fit_column
    def test_target_encoder_fit_column(self):

        df = pd.DataFrame({'variable': ['positive', 'positive', 'negative',
                                        'neutral', 'negative', 'positive',
                                        'negative', 'neutral', 'neutral',
                                        'neutral'],
                           'target': [1, 1, 0, 0, 1, 0, 0, 0, 1, 1]})

        encoder = TargetEncoder(columns=["variable"])
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

        encoder = TargetEncoder(columns=["variable"], weight=1)
        actual = encoder._fit_column(X=df.variable, y=df.target,
                                     global_mean=global_mean)

        expected = pd.Series(data=[0.375, 0.500, 0.625],
                             index=["negative", "neutral", "positive"])
        expected.index.name = "variable"

        pd.testing.assert_series_equal(actual, expected,
                                       check_less_precise=3)

    # Tests for fit method
    def test_target_encoder_fit_value_error(self):

        X = pd.DataFrame({'variable': ['positive', 'positive', 'negative']})

        target = pd.Series([1, 1, 0, 0])

        encoder = TargetEncoder(columns=["variable"])
        with pytest.raises(ValueError):
            encoder.fit(X, target)

    def test_target_encoder_fit(self):

        df = pd.DataFrame({'variable': ['positive', 'positive', 'negative',
                                        'neutral', 'negative', 'positive',
                                        'negative', 'neutral', 'neutral',
                                        'neutral'],
                           'target': [1, 1, 0, 0, 1, 0, 0, 0, 1, 1]})

        encoder = TargetEncoder(columns=["variable"])
        encoder.fit(X=df, y=df.target)

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

        expected = pd.DataFrame({'variable_enc': [0.666667, 0.666667,
                                                  0.333333, 0.50000,
                                                  0.333333, 0.666667,
                                                  0.333333, 0.50000,
                                                  0.50000, 0.50000]})

        encoder = TargetEncoder(columns=["variable"])
        encoder.fit(X=df, y=df.target)
        actual = encoder.transform(X=df, y=df.target)

        pd.testing.assert_frame_equal(actual, expected,
                                      check_less_precise=5)

    # Tests for _get_categorical_columns
    def test_target_encoder_get_categorical_columns(self):

        df = pd.DataFrame({"continuous": [1.0, 1.5, 2.0],
                           "categorical": ["negative", "neutral", "positive"],
                           "object": ["cats", "dogs", "goldfish"]})

        expected = ["categorical", "object"]

        encoder = TargetEncoder()
        actual = encoder._get_categorical_columns(df)

        # It is OK to take sets here because we also do that in the
        # _get_categorical_columns function
        assert set(actual) == set(expected)

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
