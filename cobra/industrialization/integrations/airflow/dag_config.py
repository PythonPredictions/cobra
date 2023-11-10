import os

from airflow.operators.python import ShortCircuitOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, date, timedelta

DAG_ID_ININITIAL_LOAD = "abt_initial_load"
DAG_ID_INGEST = "abt_ingest_sources"
DAG_ID_CREATE_ABT = "create_abt"
DAG_ID_BACKUP_CLEAN = "backup_and_clean_abt"

AIRFLOW_HOME = os.environ["AIRFLOW_HOME"]
DAGS_FOLDER = os.getenv("DAGS_FOLDER", os.path.join(AIRFLOW_HOME, "dags"))
BQ_SCRIPTS_FOLDER = os.path.join(DAGS_FOLDER, "scripts", "bq")


def create_next_dags(next_dag_id: str, chain_dags: str, dag: DAG, logical_date_ts: str):
    """
    Function to create the next dags, based on a chain dags condition. If the condition is set to
    the string 'true', then the next dag as in @param next_dag_id will be triggered. Otherwise, it
    will not
    :param next_dag_id: ID of the next dag to trigger if chain_dag is 'true'
    :param chain_dags: 'true' if next dag should be triggered
    :param run_date: run date in %Y%m%d or %Y-%m-%d to pass to next dag
    :param dag: instance of airflow DAG
    :return: a ShortCircuitOperator operator followed by a TriggerDagRunOperator based on chain_dags
    and next_dag_id
    """
    task_check_if_chained_dags = ShortCircuitOperator(
        task_id='check_if_chained_dags',
        op_kwargs={"do_run": chain_dags},
        python_callable=lambda do_run: do_run.lower() == "true",
        dag=dag
    )

    task_trigger_next_dag = TriggerDagRunOperator(
        task_id=f"trigger_{next_dag_id}",
        trigger_dag_id=next_dag_id,
        dag=dag,
        execution_date=logical_date_ts,
        wait_for_completion=True,  # ensure we don't run this DAG again until the following is done
        conf={
            "chain_dags": chain_dags
        }
    )

    task_check_if_chained_dags.set_downstream(task_trigger_next_dag)

    return task_check_if_chained_dags


def create_auth_task(dag: DAG):
    """
    Function to create an airflow task based on a Bash operator.
    Based on the env var "NEED_TO_GCLOUD_AUTH" being "true", the task will authenticate via
     gcloud auth. Otherwise, it will just echo 'No need to authenticate, skipping login'.
    :param dag: instance of airflow DAG
    :return: BashOperator with the right bash command to run
    """
    need_to_auth = os.getenv("NEED_TO_GCLOUD_AUTH", "False")
    if need_to_auth.lower() == "true":
        login_command = f"gcloud auth activate-service-account "\
                        f"{os.getenv('GCP_SERVICE_ACCOUNT_NAME')} "\
                        f"--key-file={os.getenv('GCP_SERVICE_ACCOUNT_KEY_PATH')}"
    else:
        login_command = "echo 'No need to authenticate, skipping login'"

    task_login = BashOperator(
        task_id="gcloud_auth_login",
        bash_command=login_command,
        dag=dag
    )

    return task_login


def get_partition_date_str(date_str, date_format="%Y%m%d"):
    run_date = datetime.strptime(date_str, date_format)

    # Partition date is the monday of the week, in %Y%m%d format
    partition_date = (run_date - timedelta(days=run_date.weekday())).strftime(date_format)
    return partition_date
