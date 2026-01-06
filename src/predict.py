# This code reads the inference input and iteratively predicts future load.

import logging
import os

import joblib
import pandas as pd
import yaml


logging.basicConfig(level=logging.INFO)

with open("inference.yaml", "r") as file:
    inference_params = yaml.safe_load(file)

with open("params.yaml", "r") as file:
    train_params = yaml.safe_load(file)["train"]


def _drop_extra_index_cols(frame):
    if "Unnamed: 0" in frame.columns:
        return frame.drop(columns=["Unnamed: 0"])
    return frame


def load_lag_values(path, lag_time):
    data = pd.read_csv(path)
    data = _drop_extra_index_cols(data)

    if "_time" in data.columns:
        data["_time"] = pd.to_datetime(data["_time"])
        data = data.sort_values("_time")
    elif "time" in data.columns:
        data["time"] = pd.to_datetime(data["time"])
        data = data.sort_values("time")

    if "netload" in data.columns:
        netload = data["netload"].astype(float).tolist()
    else:
        required = {"poiActvPwr", "pvActvPwr", "essPcsActvPwr"}
        missing = required - set(data.columns)
        if missing:
            raise ValueError(f"Missing columns for net load: {missing}")
        netload = (
            data["poiActvPwr"] + data["essPcsActvPwr"] + data["pvActvPwr"]
        ).astype(float).tolist()

    if len(netload) < lag_time:
        raise ValueError(f"Need at least {lag_time} rows for lag features")
    return netload[-lag_time:]


def load_forecast_input(path):
    data = pd.read_csv(path)
    data = _drop_extra_index_cols(data)

    if "time" not in data.columns:
        raise ValueError("Inference input must include a time column")

    if "Average_Temperature_C" not in data.columns:
        temp_col = None
        for candidate in ["Temperature_C", "temperature_2m"]:
            if candidate in data.columns:
                temp_col = candidate
                break
        if temp_col is None:
            raise ValueError("Inference input must include temperature column")
        data = data.rename(columns={temp_col: "Average_Temperature_C"})

    data["time"] = pd.to_datetime(data["time"], errors="coerce")
    data = data.dropna(subset=["time"]).sort_values("time")
    data["Average_Temperature_C"] = data["Average_Temperature_C"].astype(float)
    return data[["time", "Average_Temperature_C"]]


def build_feature_row(timestamp, temperature, lag_state):
    features = {
        "Hour": timestamp.hour,
        "Average_Temperature_C": temperature,
        "Day_of_Week": timestamp.day,
        "Month": timestamp.month,
        "DayOfYear": timestamp.year,
    }
    for i in range(1, len(lag_state) + 1):
        features[f"netload_lag{i}"] = lag_state[-i]
    return features


def iterative_predict(model, forecast_data, lag_state, feature_cols):
    predictions = []
    lag_state = list(lag_state)

    for _, row in forecast_data.iterrows():
        feature_row = build_feature_row(
            row["time"], row["Average_Temperature_C"], lag_state
        )
        feature_frame = pd.DataFrame([feature_row])[feature_cols]
        predicted = float(model.predict(feature_frame)[0])

        predictions.append(
            {
                "time": row["time"],
                "Average_Temperature_C": row["Average_Temperature_C"],
                "predicted_netload": predicted,
                **{f"netload_lag{i}": lag_state[-i] for i in range(1, len(lag_state) + 1)},
            }
        )
        lag_state.append(predicted)
        lag_state = lag_state[1:]

    return pd.DataFrame(predictions)


def load_resume_state(output_path, lag_time, lag_state, forecast_data):
    if not output_path or not os.path.exists(output_path):
        return lag_state, forecast_data, None

    existing = pd.read_csv(output_path)
    existing = _drop_extra_index_cols(existing)
    if existing.empty or "predicted_netload" not in existing.columns:
        return lag_state, forecast_data, None

    existing["time"] = pd.to_datetime(existing["time"], errors="coerce")
    existing = existing.dropna(subset=["time"]).sort_values("time")
    if existing.empty:
        return lag_state, forecast_data, None

    predicted_values = existing["predicted_netload"].astype(float).tolist()
    lag_state = list(lag_state) + predicted_values
    lag_state = lag_state[-lag_time:]

    last_time = existing["time"].iloc[-1]
    forecast_data = forecast_data[forecast_data["time"] > last_time]
    return lag_state, forecast_data, last_time


def make_predictions():
    preprocess_cfg = inference_params["preprocess_infer"]
    predict_cfg = inference_params["predict"]
    lag_time = preprocess_cfg["lag_time"]

    model = joblib.load(predict_cfg["model"])
    lag_state = load_lag_values(preprocess_cfg["input_load"], lag_time)
    forecast_data = load_forecast_input(predict_cfg["input"])
    feature_cols = preprocess_cfg.get("cols_needed", train_params["feature_cols"])
    resume_enabled = predict_cfg.get("resume", False)

    if resume_enabled:
        lag_state, forecast_data, last_time = load_resume_state(
            predict_cfg.get("output"), lag_time, lag_state, forecast_data
        )
        if last_time is not None:
            logging.info("Resuming after last predicted time: %s", last_time)

    predictions = iterative_predict(model, forecast_data, lag_state, feature_cols)
    output_path = predict_cfg["output"]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if resume_enabled and predict_cfg.get("output") and os.path.exists(predict_cfg["output"]):
        existing = pd.read_csv(output_path)
        combined = pd.concat([existing, predictions], ignore_index=True)
        combined.to_csv(output_path, index=False)
    else:
        predictions.to_csv(output_path, index=False)
    logging.info("Predictions saved to: %s", output_path)


if __name__ == "__main__":
    make_predictions()
