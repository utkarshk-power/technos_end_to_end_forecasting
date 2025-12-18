import mlflow
from urllib.parse import urlparse
import sklearn
import os
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import yaml
import logging
from sklearn.model_selection import TimeSeriesSplit

logging.basicConfig(level=logging.INFO)

with open("params.yaml", "rb") as file:
    params = yaml.safe_load(file)['train']

def prepare_train_test_data(data_path, feature_cols, 
                            target_col, train_size, val_size,
                            random_state):
    data = pd.read_csv(data_path)
    x = data[feature_cols] 
    y = data[target_col].values.ravel()
    split_idx = int(train_size * len(data))
    split_idx_val = int((train_size + val_size) * len(data))
    x_train, x_eval = x.iloc[0:split_idx], x.iloc[split_idx:split_idx_val]
    y_train, y_eval = y.iloc[0:split_idx], y.iloc[split_idx:split_idx_val]
    logging.info("Training data size: %d, Evaluation data size: %d", len(x_train), len(x_eval))
    return x_train, x_eval, y_train, y_eval

def train_model(x_train, y_train, n_estimators, max_depth, learning_rate, subsample):
    tscv = TimeSeriesSplit(n_splits=3)
    param_grid = {
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'learning_rate': learning_rate,
        'subsample': subsample
    }
    grid_search = GridSearchCV(XGBRegressor(), scoring='neg_mean_squared_error', param_grid=param_grid, cv=tscv)
    grid_search.fit(x_train, y_train)
    best_model = grid_search.best_estimator_
    return best_model, grid_search.best_params_



if __name__ == "__main__":
    os.environ['MLFLOW_TRACKING_URI'] = "https://dagshub.com/utkarshk-power/technos_end_to_end_forecasting.mlflow"
    os.environ['MLFLOW_TRACKING_USERNAME'] = "utkarshk-power"
    os.environ['MLFLOW_TRACKING_PASSWORD'] = "296f3fd6ba684d56fefb37e47706f379600fb0e5"
    mlflow.set_tracking_uri("https://dagshub.com/utkarshk-power/technos_end_to_end_forecasting.mlflow")
    mlflow.set_experiment("Technos Load Forecasting With MLOps Pipeline")
    tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
    x_train, x_eval, y_train, y_eval = prepare_train_test_data(params['data'], 
                                                             params['feature_cols'], 
                                                             params['target_col'], 
                                                             params['train_size'], 
                                                             params['val_size'],
                                                             params['random_state'])
    model, best_params = train_model(x_train, y_train, params['n_estimators'], 
                                    params['max_depth'], params['learning_rate'], 
                                    params['subsample'])
    train_predictions = model.predict(x_train)
    val_predictions = model.predict(x_eval)
    train_mse=mean_squared_error(y_train, train_predictions)
    train_mae=mean_absolute_error(y_train, train_predictions)
    train_r2=r2_score(y_train, train_predictions)
    val_mse=mean_squared_error(y_eval, val_predictions)
    val_mae=mean_absolute_error(y_eval, val_predictions)
    val_r2=r2_score(y_eval, val_predictions)
    logging.info("Training Metrics: MSE=%.4f, MAE=%.4f, R2=%.4f", train_mse, train_mae, train_r2)
    logging.info("Validation Metrics: MSE=%.4f, MAE=%.4f, R2=%.4f", val_mse, val_mae, val_r2)
    with mlflow.start_run():
        mlflow.log_params(best_params)
        mlflow.log_metrics({"train_mse": train_mse, "train_mae": train_mae, "train_r2": train_r2,
                             "val_mse": val_mse, "val_mae": val_mae, "val_r2": val_r2})
        mlflow.set_tag("model_type", "XGBoost Regressor")
        mlflow.set_tag("Developer", "Utkarsh Kulshrestha")
        mlflow.set_tag("Data Used", " Technos Data August-September 2025")
        mlflow.set_tag("Site", "Technos")
        signature = mlflow.models.infer_signature(x_train, train_predictions)
        if tracking_url_type_store != "file":
            mlflow.sklearn.log_model(sk_model=model, artifact_path="model", signature=signature, input_example=x_train.head(3))
    os.makedirs(os.path.dirname(params['model']), exist_ok=True)
    joblib.dump(model, params['model'])
    logging.info("Model saved at %s", params['model'])       


   

