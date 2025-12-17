## First End to End MLOps Powered Load Forecasting - Technos Site

1. We will try to follow standard MLOps steps for developing this Forecasting Project.
2. A separate module (.py script) for each functionality. For e.g., data loading, preprocessing, training and evaluation
3. params.yaml consists of the project dependencies.
4. Versioning of code, data and model artifacts using MLFlow, Github, DVC and Dagshub (for MLFlow server).
5. The best model for production will be chosen from MLFlow via production_critera.py.
6. A CI pipeline for unit tests, data schema tests, and smoke-level parity tests. 
7. Docker and docker-compose for containerization of the model
8. Push the inference container image to a registry and the model artifact to object storage.
9. The edge pulls the best model and installs the container along with inference.py.
10. The inference script at the edge is used to make real-time predictions for the next 2 days.
11. The predictions are passed to Prometheus and Grafana. The raw metrics are collected and visualized.
12. An Airflow DAG is triggered in case of a performance drift.
13. Upon drift, a re-training pipeline is triggered (using dvc repro) and we start again from Step 4 until model registry push.


## Revised End-to-End Flow (Minimal but Real)
## Phase 1 — Development & First Deployment
    Modular Python scripts for:
        data loading
        preprocessing
        training
        evaluation
        inference
        params.yaml for all hyperparameters and paths.
    Versioning:
        Code → GitHub
        Data → DVC
        Models & metrics → MLflow (via DagsHub)
        Model training and evaluation logged to MLflow.
        production_criteria.py selects the best model from MLflow.
    CI pipeline (GitHub Actions):
        unit tests
        data schema tests
        golden/parity tests
        smoke training (small sample only)
    Docker image built after CI passes:
        inference runtime only
        no model baked inside
    Push:
        Docker image → container registry
        Model artifact → object storage
    Edge:
        pulls Docker image
        pulls latest approved model
        runs inference service

## Phase 2 — Monitoring & Drift Detection
    Edge inference emits:
        latency
        errors
        prediction stats
        Prometheus scrapes metrics.
    Grafana:
        dashboards
        alerts (latency spikes, error rates)
        Reference + live data stored centrally.

## Phase 3 — Retraining & Continuous Improvement
    Airflow DAG (central, not edge):
        scheduled drift check (Evidently or custom)
        decision gate
    On drift:
        dvc repro
        retrain model
        evaluate
        log to MLflow
        select best model
        publish artifact
        Edge pulls updated model → cycle repeats.
