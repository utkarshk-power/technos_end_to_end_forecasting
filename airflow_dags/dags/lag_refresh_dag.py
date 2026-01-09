import airflow
from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

'''
with DAG('refresh_lag_data_dag',
         start_date=datetime(2026,1,1),
         schedule=timedelta(hours=5),
         catchup=False
) as dag:
    fetch_lag_data_task = BashOperator(task_id="fetch_lag_data", 
                                       bash_command="PYTHONPATH= . python src/fetch_lag_data.py",
                                       cwd="/Users/urkarsh.kulshrestha/Documents/AI_environment/work_env/technos_end_to_end_forecasting")
    fetch_weather_data_task = BashOperator(task_id="fetch_weather_data", 
                                           bash_command="PYTHONPATH= . python src/fetch_weather_inference.py",
                                           cwd="/Users/urkarsh.kulshrestha/Documents/AI_environment/work_env/technos_end_to_end_forecasting")
    preprocess_task = BashOperator(task_id="preprocess_data",
                                  bash_command="PYTHONPATH= . python src/preprocess.py",
                                  cwd="/Users/urkarsh.kulshrestha/Documents/AI_environment/work_env/technos_end_to_end_forecasting")
    predict_task = BashOperator(task_id = "predict",
                                bash_command="PYTHONPATH= . python src/predict.py",
                                cwd="/Users/urkarsh.kulshrestha/Documents/AI_environment/work_env/technos_end_to_end_forecasting")
    fetch_lag_data_task >> fetch_weather_data_task >> preprocess_task >> predict_task
'''
with DAG('refresh_lag_data_and_prediction_dag',
         start_date=datetime(2026,1,8),
         schedule=timedelta(minutes=1),
         catchup=False,
         default_args = { 'retries': 1, 'retry_delay': timedelta(minutes=5)},
         tags = ['forecasting', 'inference', 'docker'],
         ) as dag:
    task_env = {
        "INFLUX_BUCKET_NAME": "PXC",
        "INFLUX_TOKEN": "fxrYNTBiuBeHlAs4liTM9nh4p15AbLq_9y-mrduh6qb0lNjXnqOlng9wJJUKDvAFAkHeJ4-2Qmb0UCotfeUsjw==",
        "SORACOM_URL": "http://57-180-177-155.napter.soracom.io:15068",
        "INFLUX_ORG": "PXC",
    }
    prepare_dirs = BashOperator(
        task_id="prepare_dirs",
        bash_command="mkdir -p /usr/local/airflow/include/forecasting/data/processed /usr/local/airflow/include/forecasting/data/predictions",
        env=task_env,
    )
    update_inference_config = BashOperator(
        task_id="update_inference_config",
        bash_command="python src/update_inference_config.py",
        cwd="/usr/local/airflow/include/forecasting",
        env=task_env,
    )
    fetch_lag_data = BashOperator(
        task_id="fetch_lag_data",
        bash_command="python src/fetch_lag_data.py",
        cwd="/usr/local/airflow/include/forecasting",
        env=task_env,
    )
    fetch_weather = BashOperator(
        task_id="fetch_weather_data",
        bash_command="python src/fetch_weather_inference.py",
        cwd="/usr/local/airflow/include/forecasting",
        env=task_env,
    )
    preprocess_infer = BashOperator(
        task_id="preprocess_inference",
        bash_command="python src/preprocess_inference.py",
        cwd="/usr/local/airflow/include/forecasting",
        env=task_env,
    )
    predict = BashOperator(
        task_id="predict",
        bash_command="python src/predict.py",
        cwd="/usr/local/airflow/include/forecasting",
        env=task_env,
    )

    prepare_dirs >> update_inference_config >> fetch_lag_data >> fetch_weather >> preprocess_infer >> predict
