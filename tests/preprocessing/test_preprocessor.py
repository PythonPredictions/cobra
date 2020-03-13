from contextlib import contextmanager
import pytest
from pytest_mock import mocker

import numpy as np
import pandas as pd

from cobra.preprocessing import PreProcessor
from cobra.preprocessing import KBinsDiscretizer
from cobra.preprocessing import TargetEncoder
from cobra.preprocessing import CategoricalDataProcessor


@contextmanager
def does_not_raise():
    yield


class TestPreProcessor:

    def test_from_pipeline(self):
        pass

    def test_fit(self):
        pass

    def test_transform(self):
        pass

    def test_train_selection_validation_split(self):
        pass

    def test_get_variable_list(self):
        pass

    def test_serialize(self):
        pass

    def test_is_valid_pipeline(self):
        pass
