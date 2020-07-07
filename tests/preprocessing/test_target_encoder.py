import pytest
import pandas as pd

from cobra.preprocessing.target_encoder import TargetEncoder


class TestTargetEncoder:

    # Tests for attributes_attributes_to_dict and set_attributes_from_dict
    def test_target_encoder_attributes_to_dict(self):

        encoder = TargetEncoder()

        mapping_data = {
            "variable": {
                "negative": 0.333333,
                "neutral": 0.50000,
                "positive": 0.666667}}

        encoder._global_mean = 0.5
        encoder._mapping = mapping_data

        actual = encoder.attributes_to_dict()

        expected = {"weight": 0.0,
                    "imputation_strategy": "mean",
                    "_global_mean": 0.5,
                    "_mapping": mapping_data}

        assert actual == expected

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

        expected = data["_mapping"]

        actual = encoder._mapping

        assert actual == expected

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
