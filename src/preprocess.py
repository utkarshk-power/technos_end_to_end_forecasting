import pandas as pd
import yaml
import os
import numpy as np
import logging


logging.basicConfig(level=logging.INFO)

## Load the parameters and file paths from param.yaml
with open("params.yaml", "r") as file:
    params = yaml.safe_load(file)['preprocess']


def preprocess_(input_path, temperature_input_path, encoding, drop_rows):
    tabular_data=pd.read_csv(input_path)
    required_cols = {'poiActvPwr', 'pvActvPwr', 'essPcsActvPwr', '_time'}
    missing = required_cols - set(tabular_data.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    temperature_data = pd.read_csv(temperature_input_path,
                                   encoding=encoding['temperature']['encoding'], 
                                   skiprows=encoding['temperature']['skiprows'])
    tabular_data['_time'] = pd.to_datetime(tabular_data['_time'])
    tabular_data_updated = tabular_data.pivot_table(
        index='_time',
        values=['poiActvPwr', 'pvActvPwr', 'essPcsActvPwr'],
    aggfunc='first'  # or 'mean' if you expect duplicates
).reset_index()
    tabular_data = tabular_data_updated.sort_values(by='_time')
    tabular_data['net_load'] = tabular_data['poiActvPwr'] + tabular_data['essPcsActvPwr'] + tabular_data['pvActvPwr']
    logging.info("Length of original data: %d", len(tabular_data))
    tabular_data.drop(tabular_data.tail(drop_rows).index, inplace = True)
    logging.info("Length of data after removing last %d rows: %d", drop_rows, len(tabular_data))
    time = tabular_data['_time'].dt.strftime('%Y-%m-%d %H:%M').tolist() ## Convert datetime to string
    netload = tabular_data['net_load'].tolist()
    pcs_active_power=tabular_data['essPcsActvPwr'].tolist()
    poi_actv_power=tabular_data['poiActvPwr'].tolist()
    pv_actv_power=tabular_data['pvActvPwr'].tolist()
    return (tabular_data, 
            netload, 
            pcs_active_power, 
            poi_actv_power, 
            pv_actv_power, 
            temperature_data)

def merge_temperature_data(start_date, end_date, input_path, temperature_input_path, encoding, drop_rows):
    tabular_data, netload, pcs_active_power, poi_actv_power, pv_actv_power, temperature_data = preprocess_(input_path, temperature_input_path, encoding, drop_rows)
    temperature_data=temperature_data.iloc[:, [0,1]]
    temperature_data.columns = ["Datetime", "Temperature_C"]
    logging.info("Length of Temperature data before processing: %d", len(temperature_data))
    temperature_data["Datetime"] = pd.to_datetime(temperature_data["Datetime"], errors='coerce')
    temperature_data = temperature_data.drop(index=0)
    temperature_data = temperature_data[(temperature_data["Datetime"] >= start_date) & (temperature_data["Datetime"] <= end_date)]
    temperature = temperature_data["Temperature_C"].astype(float).tolist()
    temperature_time = temperature_data["Datetime"].dt.strftime('%Y-%m-%d %H:%M').tolist()
    if len(temperature) != len(tabular_data):
        raise ValueError("Length of Temperature Data and Plant Data do not match")
    merged_data = {"time":temperature_time, "netload":netload, "pcs_actv_power":pcs_active_power, "poi_actv_power":poi_actv_power, "pv_actv_power":pv_actv_power, "Average_Temperature_C":temperature}
    merged_dataframe = pd.DataFrame(merged_data)
    return merged_dataframe

def process_time(merged_dataframe, lag_time, lag_data):
    merged_dataframe['time'] = merged_dataframe['time'].astype(str)
    merged_dataframe['Datetime'] = pd.to_datetime(merged_dataframe['time'])
    merged_dataframe['Hour'] = merged_dataframe['Datetime'].dt.hour
    merged_dataframe['DayOfWeek'] = merged_dataframe['Datetime'].dt.day
    merged_dataframe['Month'] = merged_dataframe['Datetime'].dt.month
    merged_dataframe['DayOfYear'] = merged_dataframe['Datetime'].dt.year
    if lag_data is None:
        for i in range(1, lag_time+1):
            merged_dataframe[f'netload_lag{i}'] = merged_dataframe['netload'].shift(i)
    else:
        for i in range(1, lag_time+1):
            merged_dataframe[f"netload_lag{i}"] = np.nan   # initialize
            merged_dataframe.loc[merged_dataframe.index[0], f"netload_lag{i}"] = lag_data["netload"].iloc[-i]
    lag_cols = [f"netload_lag{i}" for i in range(1, lag_time + 1)]
    merged_dataframe = merged_dataframe.dropna(subset=lag_cols)
    logging.info("Length of data after adding time features and lag features: %d", len(merged_dataframe))
    return merged_dataframe

if __name__ == "__main__":
    merged_dataframe = merge_temperature_data(params['start_date'], params['end_date'], params['input'], params['input_temperature'], params['temperature'], params['drop_rows'])
    processed_dataframe = process_time(merged_dataframe, params['lag_time'], None)
    processed_dataframe.to_csv(params['output'], header=None, index=False)
    print("Preprocessing complete. Processed data saved to:", params['output'])
    logging.info("Preprocessing complete. Processed data saved to: %s", params['output'])







