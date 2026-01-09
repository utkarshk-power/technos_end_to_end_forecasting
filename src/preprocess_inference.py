# Preprocessing code for inference temperature data

import logging

import pandas as pd
import yaml


logging.basicConfig(level=logging.INFO)

with open("inference.yaml", "r") as file:
    inference_all = yaml.safe_load(file)
    inference_params = inference_all["preprocess_infer"]


def _drop_extra_index_cols(frame):
    if "Unnamed: 0" in frame.columns:
        return frame.drop(columns=["Unnamed: 0"])
    return frame


def load_temperature_forecast(path):
    data = pd.read_csv(path)
    data = _drop_extra_index_cols(data)

    time_col = None
    for candidate in ["Datetime", "date", "time"]:
        if candidate in data.columns:
            time_col = candidate
            break
    if time_col is None:
        raise ValueError("Temperature forecast must include a time column")

    temp_col = None
    for candidate in ["Average_Temperature_C", "Temperature_C", "temperature_2m"]:
        if candidate in data.columns:
            temp_col = candidate
            break
    if temp_col is None:
        raise ValueError("Temperature forecast must include a temperature column")

    data = data.rename(columns={time_col: "time", temp_col: "Average_Temperature_C"})
    data["time"] = pd.to_datetime(data["time"], errors="coerce", utc=True)
    data = data.dropna(subset=["time"]).sort_values("time")

    start_date = inference_params.get("temp_start_date")
    end_date = inference_params.get("temp_end_date")
    if start_date and end_date:
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)
        if start.tzinfo is None:
            start = start.tz_localize("UTC")
        else:
            start = start.tz_convert("UTC")
        if end.tzinfo is None:
            end = end.tz_localize("UTC")
        else:
            end = end.tz_convert("UTC")
        data = data[(data["time"] >= start) & (data["time"] <= end)]

    data["Average_Temperature_C"] = data["Average_Temperature_C"].astype(float)
    data["time"] = data["time"].dt.tz_localize(None)
    return data[["time", "Average_Temperature_C"]]


if __name__ == "__main__":
    forecast = load_temperature_forecast(inference_params["input_temperature"])
    forecast.to_csv(inference_params["output"], index=False)
    logging.info("Inference input saved to: %s", inference_params["output"])
