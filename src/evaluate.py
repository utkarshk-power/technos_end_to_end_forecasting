import mlflow
import pandas as pd
import joblib
import yaml
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from urllib.parse import urlparse
import logging

logging.basicConfig(level=logging.INFO)

with open("params.yaml", "rb") as file:
    params = yaml.safe_load(file)['evaluate']

def evaluate_model(data_path, feature_cols, target_col, model_path, test_size, random_state):
    data=pd.read_csv(data_path)
    x= data[feature_cols]
    y = data[target_col].values.ravel()
    split_idx = int((1 - test_size) * len(data))
    x_test = x.iloc[split_idx:]
    y_test = y.iloc[split_idx:]
    model = joblib.load(model_path)
    predictions = model.predict(x_test)
    test_mse = mean_squared_error(y_test, predictions)
    test_mae = mean_absolute_error(y_test, predictions)
    test_r2 = r2_score(y_test, predictions)
    logging.info("Test Metrics: MSE=%.4f, MAE=%.4f, R2=%.4f", test_mse, test_mae, test_r2)

    mlflow.log_metrics({
        "test_mse": test_mse,
        "test_mae": test_mae,
        "test_r2": test_r2
    })
    mlflow.set_tag("model", "XGBoostRegressor")
    mlflow.set_tag("model_stage", "evaluate")

if __name__ == "__main__":
    os.environ['MLFLOW_TRACKING_URI'] = "https://dagshub.com/utkarshk-power/technos_end_to_end_forecasting.mlflow"
    os.environ['MLFLOW_TRACKING_USERNAME'] = "utkarshk-power"
    os.environ['MLFLOW_TRACKING_PASSWORD'] = "296f3fd6ba684d56fefb37e47706f379600fb0e5"
    mlflow.set_tracking_uri("https://dagshub.com/utkarshk-power/technos_end_to_end_forecasting.mlflow")
    mlflow.set_experiment("Technos Load Forecasting With MLOps Pipeline")
    evaluate_model(params['data'], 
                   params['feature_cols'], 
                   params['target_col'], 
                   params['model'], 
                   params['test_size'], 
                   params['random_state'])

