import pandas as pd
from src.train import train_model
import os
import joblib

# Check if model file is created after training

def test_model_file_created():
    model_path = 'models/model.pkl'
    assert os.path.exists(model_path), "Model file was not created"
    model = joblib.load(model_path)
    assert model is not None, "Loaded model is None"


def test_check_lag_file_exists():
    lag_file_path = 'data/processed/inference_lag_data.csv'
    assert os.path.exists(lag_file_path), "Lag data file for inference does not exist"
    lag_data = pd.read_csv(lag_file_path)
    assert not lag_data.empty, "Lag data file is empty"
    expected_columns = ["_time", "poiActvPwr", "pvActvPwr", "essPcsActvPwr"]
    for col in expected_columns:
        assert col in lag_data.columns, f"Expected column '{col}' not found in lag data"
        assert len(lag_data) >= 5, "Lag data does not contain enough rows for lag features"

def test_check_temperature_file_exists():
    temperature_file_path = 'data/processed/aichi_temperature_forecast.csv'
    assert os.path.exists(temperature_file_path), "Temperature data file does not exist"
    temperature_data = pd.read_csv(temperature_file_path)
    assert not temperature_data.empty, "Temperature data file is empty"


def test_check_inference_input_file_exists():
    inference_input_path = 'data/processed/inference_input_data.csv'
    assert os.path.exists(inference_input_path), "Inference input data file does not exist"
    inference_data = pd.read_csv(inference_input_path)
    assert not inference_data.empty, "Inference input data file is empty"


