import pandas as pd
from src.model.common.CobraScorer import CobraScorer, CobraModelSerializationType
from src.model.common.BQCobraScorer import BQCobraScorer
from google.cloud import bigquery
from google.oauth2 import service_account

if __name__ == '__main__':
    postcodes_path = '/Users/nicolas.morandi/workspace/brico-analytical-base-tables/local/data/Verrijkte_postcodes.xlsx'
    # model_path = '/Users/nicolas.morandi/workspace/brico-analytical-base-tables/local/data/forward_selection_garden_project.pickle'
    # pipeline_path = '/Users/nicolas.morandi/workspace/brico-analytical-base-tables/local/data/preprocessing_pipeline_garden_project.json'

    pipeline_path = 'gs://garden-model-stg/20220407/preprocessing_pipeline_garden_project.json'
    model_path = 'gs://garden-model-stg/20220407/forward_selection_garden_project.pickle'
    key_path = '/Users/nicolas.morandi/workspace/brico-analytical-base-tables/local/keys/analytical-base-tables-staging-sa_api.json'
    run_date = '20220203'
    #
    # credentials = service_account.Credentials.from_service_account_file(
    #     key_path, scopes=["https://www.googleapis.com/auth/cloud-platform"],
    # )
    #
    # client = bigquery.Client(credentials=credentials, project=credentials.project_id, location='eu')


    # change select * to columns
    # query = """
    #     SELECT *
    #     FROM `analytical-base-tables-staging.predictive_test.garden_score_with_postcodes`
    # """
    # query_job = client.query(query)
    #
    # basetable_model = query_job.result().to_dataframe()

    cont_vars = ['lifetime_sales_amount', 'sum_mon_dep_tuingereedschap',
                      'sum_email_click_count_7d', 'ratio_mail_click_delivered_Q2',
                      'ratio_mail_click_delivered_Q3', 'mon_garden_March']

    cat_vars = ['NEWHABI']
    # scorer = CobraScorer(pipeline_path=pipeline_path,
    #                      model_path=model_path,
    #                      model_serialization_type=CobraModelSerializationType.PICKLE,
    #                      continuous_vars=cont_vars,
    #                      discrete_vars=cat_vars,
    #                      id_column_name='user_id',
    #                      score_column_name='score',
    #                      step=5
    #                      )
    #
    # scores = scorer.score(basetable_model)
    #
    # print(scores)

    # cobra_scorer = BQCobraScorer(input_table="analytical-base-tables-staging.predictive_test.garden_score_with_postcodes_small",
    #                              output_table="analytical-base-tables-staging.output_devnm.garden_score",
    #                              pipeline_path=pipeline_path,
    #                              model_path=model_path,
    #                              model_serialization_type=CobraModelSerializationType.PICKLE,
    #                              continuous_vars=cont_vars,
    #                              discrete_vars=cat_vars,
    #                              id_column_name='user_id',
    #                              score_column_name='score',
    #                              date_partition_column_name='score_date',
    #                              key_path=key_path,
    #                              location='eu',
    #                              step=5
    #                              )
    # df_scores = cobra_scorer.score()
    # print(df_scores)
    # cobra_scorer.write_scores_to_bq(df_scores, run_date)

    BQCobraScorer.score_and_save(score_date=run_date,
                                 input_table="analytical-base-tables-staging.predictive_test.garden_score_with_postcodes_small",
                                 output_table="analytical-base-tables-staging.output_devnm.garden_score",
                                 pipeline_path=pipeline_path,
                                 model_path=model_path,
                                 model_serialization_type=CobraModelSerializationType.PICKLE,
                                 continuous_vars=cont_vars,
                                 discrete_vars=cat_vars,
                                 id_column_name='user_id',
                                 score_column_name='score',
                                 date_partition_column_name='score_date',
                                 key_path=key_path,
                                 location='eu',
                                 step=5
                                 )
