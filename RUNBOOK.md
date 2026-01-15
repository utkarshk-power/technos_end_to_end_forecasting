Edge Airflow Runbook (Astro Image + Mounted DAGs)
=================================================

Goal
----
Run Airflow on the edge (Docker, Astro-based image), mount DAGs for fast iteration,
mount output/log folders, and access the UI from the Mac via SSH tunnel.

What Is Baked vs Mounted
------------------------
Baked into image:
- airflow_dags/include/forecasting (scripts, configs, model)
- airflow_dags/requirements.txt (Python deps for tasks)

Mounted from edge:
- DAGs folder (edit without rebuild)
- forecasting folder (writable inference.yaml)
- processed, predictions, logs folders

Repository Layout (Mac)
-----------------------
technos_end_to_end_forecasting/
  airflow_dags/
    Dockerfile
    requirements.txt
    dags/
      refresh_prediction_local.py
      lag_refresh_dag.py
      refresh_predictions_ssh.py
    include/
      forecasting/
        inference.yaml
        params.yaml
        models/
        src/

Edge Layout
-----------
/opt/
  airflow/
    .env
    dags_admin/
    forecasting/
    processed/
    predictions/
    logs_admin/
  technos_end_to_end_forecasting/   # kept for SSH DAGs

Dockerfile (current)
--------------------
FROM astrocrpublic.azurecr.io/runtime:3.1-9
COPY include/ /usr/local/airflow/include/
COPY requirements.txt /usr/local/airflow/requirements.txt
RUN pip install --no-cache-dir -r /usr/local/airflow/requirements.txt

Build + Copy (Mac -> Edge)
--------------------------
1) Build amd64 image on Mac:
   docker buildx build --platform linux/amd64 -t technos-airflow:edge ./airflow_dags

2) Save + copy to edge:
   docker save technos-airflow:edge | gzip > /tmp/technos-airflow-edge.tar.gz
   scp /tmp/technos-airflow-edge.tar.gz admin@192.168.2.10:/opt/

3) Load image on edge:
   docker load -i /opt/technos-airflow-edge.tar.gz

Prepare Edge Folders + Permissions
----------------------------------
mkdir -p /opt/airflow/dags_admin /opt/airflow/forecasting \
         /opt/airflow/processed /opt/airflow/predictions /opt/airflow/logs_admin
chmod -R 777 /opt/airflow/processed /opt/airflow/predictions /opt/airflow/logs_admin /opt/airflow/forecasting

Copy DAGs and Forecasting Folder
--------------------------------
Copy DAGs (from Mac):
rsync -av /Users/urkarsh.kulshrestha/Documents/AI_environment/work_env/technos_end_to_end_forecasting/airflow_dags/dags/ \
  admin@192.168.2.10:/opt/airflow/dags_admin/

Copy forecasting (from Mac):
rsync -av /Users/urkarsh.kulshrestha/Documents/AI_environment/work_env/technos_end_to_end_forecasting/airflow_dags/include/forecasting/ \
  admin@192.168.2.10:/opt/airflow/forecasting/

Edge .env (Required)
--------------------
Create /opt/airflow/.env with:
INFLUX_BUCKET_NAME=...
INFLUX_TOKEN=...
INFLUX_ORG=PXC
SORACOM_URL=...   # only used if scripts read env directly

AIRFLOW__WEBSERVER__SECRET_KEY=<openssl rand -hex 32>
AIRFLOW__API__AUTH_JWT_SECRET=<openssl rand -hex 32>
AIRFLOW__CORE__DAGS_FOLDER=/usr/local/airflow/dags
AIRFLOW__DAG_PROCESSOR__ENABLED=True

Run Container (Edge)
--------------------
docker rm -f refresh-predictions-edge 2>/dev/null || true

docker run -d --name refresh-predictions-edge \
  -p 8080:8080 \
  --env-file /opt/airflow/.env \
  -v /opt/airflow/dags_admin:/usr/local/airflow/dags \
  -v /opt/airflow/forecasting:/usr/local/airflow/include/forecasting \
  -v /opt/airflow/processed:/usr/local/airflow/include/forecasting/data/processed \
  -v /opt/airflow/predictions:/usr/local/airflow/include/forecasting/data/predictions \
  -v /opt/airflow/logs_admin:/usr/local/airflow/logs \
  technos-airflow:edge \
  airflow standalone

UI Access (Mac via Tunnel)
--------------------------
ssh -N -L 8083:192.168.2.10:8080 admin@192.168.2.10
Open http://localhost:8083

Get UI password (edge):
docker logs refresh-predictions-edge | grep -i "Password for user 'admin'"

Critical Fixes (Issues + Remedies)
----------------------------------
1) No DAGs in UI
   - Cause: DAG processor couldn't write logs (PermissionError).
   - Fix: chmod -R 777 /opt/airflow/logs_admin

2) PermissionError on inference.yaml
   - Cause: file inside image is read-only.
   - Fix: mount /opt/airflow/forecasting to /usr/local/airflow/include/forecasting

3) PermissionError on processed/predictions CSVs
   - Fix: chmod -R 777 /opt/airflow/processed /opt/airflow/predictions

4) Login loops / 403
   - Cause: unstable secrets or stale cookies.
   - Fix: set AIRFLOW__WEBSERVER__SECRET_KEY and AIRFLOW__API__AUTH_JWT_SECRET in .env, restart, use incognito.

5) Soracom DNS failures
   - Scripts read lag_data.url from inference.yaml (not .env).
   - Fix: update lag_data.url to IP (with http://) in /opt/airflow/forecasting/inference.yaml

Update inference.yaml URL (Edge)
--------------------------------
sed -i 's|^  url:.*|  url: http://18.181.70.247:43321|' /opt/airflow/forecasting/inference.yaml
grep -n "url:" /opt/airflow/forecasting/inference.yaml

Rebuild vs Copy Rules
---------------------
Rebuild image when:
- airflow_dags/include/forecasting changes
- airflow_dags/requirements.txt changes

Only rsync DAGs when:
- airflow_dags/dags/* changes
