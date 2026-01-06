#!/bin/sh
set -e

python src/fetch_lag_data.py
python src/fetch_weather_inference.py
python src/preprocess_inference.py
python src/predict.py
