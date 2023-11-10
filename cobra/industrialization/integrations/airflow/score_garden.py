from datetime import datetime

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.google.cloud.operators.bigquery import BigQueryInsertJobOperator

from ABTTablesConfig import ABTTablesConfig
import dag_config as conf
from BQCobraScorer import BQCobraScorer
from CobraScorer import CobraModelSerializationType

# DAG to pre process and score the garden prediction model
# With a few changes (see TODOs) the DAG could be generic for any model or a loop over
# configurations could dynamically generate more than 1 DAG

dryRun = False

abtConf = ABTTablesConfig()

dict_table_names = abtConf.get_target_table_names()

dag = DAG(
    dag_id="score_garden_table",
    start_date=datetime(year=2021, month=1, day=1),
    # end_date=datetime(year=2022, month=3, day=20),
    default_args={"email_on_failure": False},
    description="DAG to run the queries necessary to populate the ABT table",
    schedule_interval=None,
    catchup=False,
    render_template_as_native_obj=False,
    max_active_runs=1,
    user_defined_macros={"input_dataset": abtConf.input_dataset,
                         "dataset_intermediate": abtConf.dataset_intermediate,
                         "dataset_output": abtConf.dataset_output,
                         "project": abtConf.target_project,
                         "bucket_backups": abtConf.bucket_backups,
                         "location": abtConf.location,
                         "run_date_format": f"\"{abtConf.run_date_format}\"",
                         "partition_date": conf.get_partition_date_str,
                         **dict_table_names  # append table names
                         }
)

# sorted list of queries to run
query_files = [
    "preprocessing.sql"
]

with dag:

    tasks = []
    for idx, query in enumerate(query_files):
        task_run_query = BigQueryInsertJobOperator(
            task_id=f"run_query_{query}",
            location=abtConf.location,
            configuration={
                "query": {
                    "query": f"{{% include 'queries/garden_pre_processing/{query}' %}}",  # path could be more generic
                    "useLegacySql": False,
                    "dryRun": dryRun
                },
            },
            dag=dag
        )

        tasks.append(task_run_query)
        if idx > 0:
            task_run_query.set_upstream(tasks[idx-1])

    # Finally, score the model
    model_name = "garden_prediction"

    task_score_model = PythonOperator(
        task_id=f"score_model_{model_name}",
        python_callable=BQCobraScorer.score_and_save,
        op_kwargs={
            'score_date': '{{ds_nodash}}',
            'input_table': "analytical-base-tables-staging.predictive_test.garden_score_with_postcodes_small",  # remove hardcoded
            'output_table': "analytical-base-tables-staging.output_devnm.garden_score",  # remove hardcoded
            'pipeline_path': 'gs://garden-model-stg/20220407/preprocessing_pipeline_garden_project.json',  # read from env?
            'model_path': 'gs://garden-model-stg/20220407/forward_selection_garden_project.pickle',  # read from env?
            'model_serialization_type': CobraModelSerializationType.PICKLE,
            'continuous_vars': ['lifetime_sales_amount', 'sum_mon_dep_tuingereedschap',
                                'sum_email_click_count_7d', 'ratio_mail_click_delivered_Q2',
                                'ratio_mail_click_delivered_Q3', 'mon_garden_March'],
            'discrete_vars': ['NEWHABI'],
            'id_column_name': 'user_id',
            'score_column_name': 'score',
            'date_partition_column_name': 'score_date',
            'location': 'eu',
            'step': 5
        },
        dag=dag
    )

    tasks[-1].set_downstream(task_score_model)
