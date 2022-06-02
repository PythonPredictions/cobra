import json
import pickle
from enum import Enum
from enum import IntEnum
from typing import List, Union, Optional

import numpy as np
from cobra.preprocessing import PreProcessor
from pandas import DataFrame
from smart_open import open

from cobra.model_building import LogisticRegressionModel, LinearRegressionModel, \
    ForwardFeatureSelection


class CobraModelSerializationType(IntEnum):
    JSON = 1
    PICKLE = 2


class CobraScorer:
    """
    Wrapper around a Cobra model that needs to be scored with data from Big Query
    and scores to be stored on Big Query
    """

    def __init__(self,
                 pipeline_path: str,
                 model_path: str,
                 model_serialization_type: CobraModelSerializationType,
                 continuous_vars: List[str],
                 discrete_vars: List[str],
                 id_column_name='id',
                 score_column_name='score',
                 **kwargs):
        """

        :param pipeline_path: path to a json file that represents a cobra PreProcessor
        :param model_path: path to a json or pickle file with a Cobra Model or Forward Selection
        :param model_serialization_type: Type of serialization used for the file in model_path
        :param continuous_vars: list of continuous variables to use when scoring the model
        :param discrete_vars: list of discrete variables to use when scoring the model
        :param id_column_name: name of the column used to identify rows in the observations and
        scores dataframes
        :param score_column_name: name of the column with scores in the output dataframe
        :param kwargs: other generic arguments, such as 'step' for forward selection
        """

        # TODO replace by base class once implemented in Cobra
        self.model: Optional[Union[LinearRegressionModel, LogisticRegressionModel]] = None
        self.preprocessor: Optional[PreProcessor] = None

        self.id_column_name = id_column_name
        self.score_column_name = score_column_name
        self.model_serialization_type = model_serialization_type
        self.continuous_vars = continuous_vars
        self.discrete_vars = discrete_vars

        self.load_pipeline(pipeline_path)
        self.load_model(model_path=model_path, **kwargs)

    @classmethod
    def deserialize_model(cls, model_dict: dict)\
            -> Union[LinearRegressionModel, LogisticRegressionModel]:
        """
        Method to deserialize a Cobra model based on a json file
        Fails if the json file does not contain a key meta with a valid model type
        TODO build as part as Cobra's model base class deserialize
        TODO replace return type when base class is created
        :param model_dict: dictionary representing the serialized model
        :return:
        """
        # TODO build as part as Cobra's model base class deserialize
        # dictionary of (meta attribute in json file), (class for that description)
        MODEL_META = {
            "linear-regression": LinearRegressionModel,
            "logistic-regression": LogisticRegressionModel
        }

        model_cls = MODEL_META.get(model_dict["meta"])

        model = model_cls()
        model.deserialize(model_dict)

        return model

    def load_pipeline(self, pipeline_path: str):
        """
        Method to load a pipeline into the preprocessor attribute
        :param pipeline_path: Path to a json file pre processing pipeline serialized as
        Supports locations supported by smart_open
        :return:
        """
        with open(pipeline_path) as pipeline_file:
            processing_pipeline = json.load(pipeline_file)
            self.preprocessor = PreProcessor.from_pipeline(processing_pipeline)

    def load_model(self, model_path: str, **kwargs):
        """
        Load a Cobra model from a json file or from a pickle file

        If the stored file represents an instance of ForwardFeatureSelection, then 'step' must be
        provided
        :param model_path:
        :param kwargs:
        :return: nothing. Loaded model is stored in the model attribute.
        """
        model = None
        if self.model_serialization_type == CobraModelSerializationType.JSON:
            with open(model_path, "r") as model_file:
                model_dict = json.load(model_file)
                model = self.deserialize_model(model_dict)

        elif self.model_serialization_type == CobraModelSerializationType.PICKLE:
            with open(model_path, "rb") as model_file:
                model = pickle.load(model_file)
        else:
            raise ValueError(f"Invalid CobraModelSerializationType: {self.model_serialization_type}")

        if isinstance(model, ForwardFeatureSelection):
            step = kwargs.get('step')
            model = model.get_model_from_step(step)

        self.model = model

    def score(self, observations: DataFrame) -> DataFrame:
        """
        Method to score a set of observations which have not been processed by a Cobra PreProcessor
        yet
        :param observations: dataframe with observations
        :return: dataframe with scores, with columns self.id_column_name and self.score_column_name
        """
        pre_processed_obs = self.preprocessor.transform(observations,
                                                        continuous_vars=self.continuous_vars,
                                                        discrete_vars=self.discrete_vars)

        return self._do_score(pre_processed_obs)

    def _do_score(self, pre_processed_obs: DataFrame) -> DataFrame:
        """
        Internal method to score a dataframe containing observations which have been processed
        by a Cobra PreProcessor
        precondition id_column_name must exist in the pre_processed_obs
        :param pre_processed_obs:
        :return: dataframe with scores, with columns self.id_column_name and self.score_column_name
        """
        scores = pre_processed_obs[[self.id_column_name]].copy()
        scores[self.score_column_name] = self.model.score_model(pre_processed_obs)

        return scores
