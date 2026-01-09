#!/bin/sh
set -e

mkdir -p data/processed data/predictions

python src/update_inference_config.py
python src/fetch_lag_data.py
python src/fetch_weather_inference.py
python src/preprocess_inference.py
python src/predict.py
