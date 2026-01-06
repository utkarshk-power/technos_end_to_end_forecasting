# We will write tests for preprocess.py
# To check whether the input files are being read correctly, not empty and
# Whether expected columns are present

import pandas as pd
from src.preprocess import preprocess_, merge_temperature_data, process_time
import os


def test_process_function_produces_output():
    processed_data = pd.read_csv('data/processed/processed_data.csv')
    assert not processed_data.empty, "Processed data should not be empty. File not found or empty."
    
def test_expected_columns_in_processed_data():
    processed_data = pd.read_csv('data/processed/processed_data.csv')
    expected_columns = ["Hour",
                       "Average_Temperature_C",
                       "Day_of_Week",
                       "Month",
                       "DayOfYear",
                       "netload_lag1",
                       "netload_lag2",
                       "netload_lag3",
                       "netload_lag4",
                       "netload_lag5"]
    for col in expected_columns:
        assert col in processed_data.columns, f"Expected column '{col}' not found in processed data"


    
