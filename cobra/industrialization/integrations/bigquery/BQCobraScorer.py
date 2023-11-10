import pandas_gbq
import pandas as pd
import json
from google.cloud import bigquery
from google.oauth2 import service_account
from typing import List, Optional

from datetime import datetime

from model.common.CobraScorer import CobraScorer, CobraModelSerializationType


class BQCobraScorer:

    def __init__(self, input_table: str,
                 output_table: str,
                 pipeline_path: str,
                 model_path: str,
                 model_serialization_type: CobraModelSerializationType,
                 continuous_vars: List[str],
                 discrete_vars: List[str],
                 id_column_name: str = 'id',
                 id_column_type: str = 'INTEGER',
                 score_column_name: str = 'score',
                 score_column_type: str = 'FLOAT',
                 date_partition_column_name: str = 'score_date',
                 key_path: Optional[str] = None,
                 location: str = 'eu',
                 audit_table: Optional[str] = None,
                 **kwargs):
        """

        :param input_table:
        :param output_table:
        :param pipeline_path:
        :param model_path:
        :param model_serialization_type:
        :param continuous_vars:
        :param discrete_vars:
        :param id_column_name:
        :param id_column_type:
        :param score_column_name:
        :param score_column_type:
        :param date_partition_column_name:
        :param key_path:
        :param location:
        :param audit_table:
        :param kwargs:
        """

        self.cobra_scorer = CobraScorer(pipeline_path=pipeline_path,
                                        model_path=model_path,
                                        model_serialization_type=model_serialization_type,
                                        continuous_vars=continuous_vars,
                                        discrete_vars=discrete_vars,
                                        id_column_name=id_column_name,
                                        score_column_name=score_column_name,
                                        **kwargs)
        self.model_path = model_path
        self.model_serialization_type = model_serialization_type
        self.continuous_vars = continuous_vars,
        self.discrete_vars = discrete_vars,
        self.pipeline_path = pipeline_path
        self.input_table = input_table
        self.output_table = output_table
        self.key_path = key_path
        self.location = location
        self.credentials = None
        self.id_column_name = id_column_name
        self.id_column_type = id_column_type
        self.score_column_name = score_column_name
        self.score_column_type = score_column_type
        self.date_partition_column_name = date_partition_column_name
        self.audit_table = audit_table if audit_table else f"{self.output_table}_audit"
        self.kwargs = kwargs

        if key_path:
            self.credentials = service_account.Credentials.from_service_account_file(
                key_path, scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )
            self.client = bigquery.Client(credentials=self.credentials,
                                          project=self.credentials.project_id,
                                          location=location)
        else:
            self.client = bigquery.Client(location=location)

    def write_scores_to_bq(self, scores: pd.DataFrame, score_date_str: str,
                           date_format: str = "%Y%m%d", overwrite_partition: bool = True):
        """
        Method that writes scores to a big query table and writes audit information into the audit
        table

        self.output table must exist and contain a column self.date_partition_column_name
        :param overwrite_partition:
        :param scores:
        :param score_date_str:
        :param date_format:
        :return:
        """
        score_date = datetime.strptime(score_date_str, date_format)
        scores[self.date_partition_column_name] = score_date
        table_schema = [{'name': self.id_column_name, 'type': self.id_column_type},
                        {'name': self.score_column_name, 'type': self.score_column_type},
                        {'name': self.date_partition_column_name, 'type': 'DATE'},
                        ]

        # Workaround due to pandas_gbq not supporting write disposition
        # https://github.com/googleapis/python-bigquery-pandas/issues/118
        if overwrite_partition:
            self._delete_partition_from_output_table(score_date)

        pandas_gbq.to_gbq(dataframe=scores,
                          destination_table=self.output_table,
                          if_exists='append',
                          credentials=self.credentials,
                          table_schema=table_schema
                          )

        self._log_score_run(score_date, overwrite_partition=overwrite_partition)

    def _delete_partition_from_output_table(self, score_date: datetime):
        self.client.delete_table(table=f'{self.output_table}${score_date.strftime("%Y%m%d")}')

    def _log_score_run(self, run_date: datetime, **kwargs):
        """
        Method that logs a run into a big query table, including
        - score_date: date for which the scores are valid (aka logical_date)
        - model_version: model version or path to the model used for scoring
        - pipeline version
        - exec_date: actual date in which the score was executed
        - extra information that might be relevant in the future
        :param run_date: date for which the model was scored
        :param kwargs: extra arguments to be logged. Must be json serializable
        :return: -
        """
        # Log all attributes except for client, cobra_scorer and credentials
        extra_args = {'model_serialization_type': self.model_serialization_type,
                      'continuous_vars': self.continuous_vars,
                      'discrete_vars': self.discrete_vars,
                      'input_table': self.input_table,
                      'output_table': self.output_table,
                      'key_path': self.key_path,
                      'location': self.location,
                      'id_column_name': self.id_column_name,
                      'id_column_type': self.id_column_type,
                      'score_column_name': self.score_column_name,
                      'score_column_type': self.score_column_type,
                      'date_partition_column_name': self.date_partition_column_name,
                      'audit_table': self.audit_table}

        if 'step' in self.kwargs:
            extra_args['step'] = self.kwargs['step']
        extra = {**extra_args, **kwargs}

        audit_df = pd.DataFrame.from_dict(
            {
                "exec_date": [datetime.now()],
                "run_date": [run_date],
                "model_path": [self.model_path],
                "pipeline_path": [self.pipeline_path],
                "extra": [json.dumps(extra)]
            }
        )

        table_schema = [{'name': 'exec_date', 'type': 'DATETIME'},
                        {'name': 'run_date', 'type': 'DATE'},
                        {'name': 'model_path', 'type': 'STRING'},
                        {'name': 'pipeline_path', 'type': 'STRING'},
                        {'name': 'extra', 'type': 'STRING'}]  # change to JSON when out of preview

        pandas_gbq.to_gbq(dataframe=audit_df,
                          destination_table=self.audit_table,
                          if_exists='append',
                          credentials=self.credentials,
                          table_schema=table_schema
                          )

    def load_observations(self) -> pd.DataFrame:
        """
        Loads observations by selecting all columns and rows from the input table
        :return: pandas dataframe with observations
        """
        query = f"SELECT * FROM `{self.input_table}`"

        return pandas_gbq.read_gbq(query, credentials=self.credentials)

    def score(self) -> pd.DataFrame:
        obs = self.load_observations()
        return self.cobra_scorer.score(observations=obs)

    @classmethod
    def score_and_save(cls,
                       score_date: str,
                       input_table: str,
                       output_table: str,
                       pipeline_path: str,
                       model_path: str,
                       model_serialization_type: CobraModelSerializationType,
                       continuous_vars: List[str],
                       discrete_vars: List[str],
                       id_column_name: str = 'id',
                       id_column_type: str = 'INTEGER',
                       score_column_name: str = 'score',
                       score_column_type: str = 'FLOAT',
                       date_partition_column_name: str = 'score_date',
                       key_path: Optional[str] = None,
                       location: str = 'eu',
                       audit_table: Optional[str] = None,
                       **kwargs
                       ):
        """
        Class method to load, score a model, and save the results to BQ in one go without needing
        to instantiate an instance. Useful for PythonOperator
        :param score_date: string representing the date for which the scores are computed
        :param input_table:
        :param output_table:
        :param pipeline_path:
        :param model_path:
        :param model_serialization_type:
        :param continuous_vars:
        :param discrete_vars:
        :param id_column_name:
        :param id_column_type:
        :param score_column_name:
        :param score_column_type:
        :param date_partition_column_name:
        :param key_path:
        :param location:
        :param audit_table:
        :param kwargs:
        :return:
        """

        cobra_scorer = BQCobraScorer(
            input_table=input_table,
            output_table=output_table,
            pipeline_path=pipeline_path,
            model_path=model_path,
            model_serialization_type=model_serialization_type,
            continuous_vars=continuous_vars,
            discrete_vars=discrete_vars,
            id_column_name=id_column_name,
            id_column_type=id_column_type,
            score_column_name=score_column_name,
            score_column_type=score_column_type,
            date_partition_column_name=date_partition_column_name,
            key_path=key_path,
            location=location,
            audit_table=audit_table,
            **kwargs
            )

        df_scores = cobra_scorer.score()
        cobra_scorer.write_scores_to_bq(df_scores, score_date)