from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta
from src.model_development import (
    load_data,
    data_preprocessing,
    build_model,
    evaluate_model
)
from src.success_email import send_success_email
from airflow import configuration as conf

# Enable pickle support for XCom
conf.set('core', 'enable_xcom_pickling', 'True')

# Default arguments
default_args = {
    'owner': 'Malav Patel',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Create DAG
dag = DAG(
    'healthcare_ml_pipeline',
    default_args=default_args,
    description='Cardiovascular Disease Prediction using Random Forest',
    schedule_interval=None,  # Manual trigger only
    catchup=False,
    tags=['healthcare', 'machine-learning', 'random-forest']
)

# Task 1: Load Data
load_data_task = PythonOperator(
    task_id='load_data',
    python_callable=load_data,
    dag=dag
)

# Task 2: Preprocess Data
preprocess_task = PythonOperator(
    task_id='preprocess_data',
    python_callable=data_preprocessing,
    op_args=[load_data_task.output],
    dag=dag
)

# Task 3: Separate Data Outputs (for XCom)
def separate_data_outputs(**kwargs):
    """Extract preprocessing outputs from XCom."""
    ti = kwargs['ti']
    X_train, X_test, y_train, y_test = ti.xcom_pull(task_ids='preprocess_data')
    return X_train, X_test, y_train, y_test

separate_task = PythonOperator(
    task_id='separate_data',
    python_callable=separate_data_outputs,
    provide_context=True,
    dag=dag
)

# Task 4: Build and Train Model
train_model_task = PythonOperator(
    task_id='train_model',
    python_callable=build_model,
    op_args=[separate_task.output, "cardiovascular_rf_model.pkl"],
    dag=dag
)

# Task 5: Evaluate Model
evaluate_task = PythonOperator(
    task_id='evaluate_model',
    python_callable=evaluate_model,
    op_args=[separate_task.output, "cardiovascular_rf_model.pkl"],
    dag=dag
)

# Task 6: Send Success Email
email_task = PythonOperator(
    task_id='send_success_email',
    python_callable=send_success_email,
    provide_context=True,
    dag=dag
)

# Define task dependencies
load_data_task >> preprocess_task >> separate_task >> train_model_task >> evaluate_task >> email_task