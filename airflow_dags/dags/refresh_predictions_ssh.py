from airflow import DAG
from airflow.providers.ssh.operators.ssh import SSHOperator
from datetime import datetime, timedelta
import os
import shlex

with DAG('refresh_predictions_edge_ssh',
         start_date=datetime(2026, 1, 10),
         schedule=timedelta(minutes=10),
         catchup=False,
         default_args={'retries':1, 'retry_delay': timedelta(minutes=5)},
         tags = ['refresh_forecasts', 'ssh', 'edge_device'],
         ) as dag:
    task_env = {
        "INFLUX_BUCKET_NAME": os.getenv("INFLUX_BUCKET_NAME", ""),
        "INFLUX_TOKEN": os.getenv("INFLUX_TOKEN", ""),
        "SORACOM_URL": os.getenv("SORACOM_URL", ""),
        "INFLUX_ORG": os.getenv("INFLUX_ORG", ""),
    }
    env_prefix = (
        "INFLUX_BUCKET_NAME={INFLUX_BUCKET_NAME} "
        "INFLUX_TOKEN={INFLUX_TOKEN} "
        "SORACOM_URL={SORACOM_URL} "
        "INFLUX_ORG={INFLUX_ORG}"
    ).format(**{k: shlex.quote(v) for k, v in task_env.items()})
    remote_base = "/opt/technos_end_to_end_forecasting"
    python_bin = f"{remote_base}/.venv/bin/python"

    ssh_task = SSHOperator(
        task_id = "execute_ssh_refresh_prediction",
        ssh_conn_id = "edge_device_ssh",
        command = f"{env_prefix} echo Hello from Utkarsh MAC, connection established",
        cmd_timeout=30
    )

    update_inference_config = SSHOperator(
        task_id="update_inference_config_ssh",
        ssh_conn_id="edge_device_ssh",
        command=f"cd {remote_base} && {env_prefix} {python_bin} src/update_inference_config.py",
        cmd_timeout=300
    )

    fetch_lag_data = SSHOperator(
        task_id="fetch_lag_data_ssh",
        ssh_conn_id="edge_device_ssh",
        command=f"cd {remote_base} && {env_prefix} {python_bin} src/fetch_lag_data.py",
        cmd_timeout=300
    )

    fetch_weather = SSHOperator(
        task_id = "fetch_weather_data_ssh",
        ssh_conn_id = "edge_device_ssh",
        command=f"cd {remote_base} && {env_prefix} {python_bin} src/fetch_weather_inference.py",
        cmd_timeout=300
    )

    preprocess_infer = SSHOperator(
        task_id="preprocess_infer_ssh",
        ssh_conn_id="edge_device_ssh",
        command=f"cd {remote_base} && {env_prefix} {python_bin} src/preprocess_inference.py",
        cmd_timeout=300
    )

    predict = SSHOperator(
        task_id = "predict_ssh",
        ssh_conn_id = "edge_device_ssh",
        command=f"cd {remote_base} && {env_prefix} {python_bin} src/predict.py",
        cmd_timeout=300
    )

    ssh_task >> update_inference_config >> fetch_lag_data >> fetch_weather >> preprocess_infer >> predict
