"""Airflow DAG to train a model."""

from datetime import datetime

from airflow import DAG
from airflow.operators.python import PythonOperator
from src.modeling.load_data import load_data
from src.modeling.train import train

default_args = {
    "owner": "airflow",
    "start_date": datetime.today(),
    "retries": 1,
}

with DAG(
    dag_id="train_model",
    description="Train a DecisionTreeClassifier model",
    default_args=default_args,
    schedule_interval="@daily",
    tags=["training"],
    catchup=False,
) as dag:
    dag_load_data = PythonOperator(
        task_id="load_data",
        python_callable=load_data,
    )

    dag_train = PythonOperator(
        task_id="train",
        python_callable=train,
        op_args=dag_load_data.output,
    )

    dag_load_data >> dag_train
