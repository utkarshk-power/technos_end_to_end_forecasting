FROM astrocrpublic.azurecr.io/runtime:3.1-9

COPY include/ /usr/local/airflow/include/
COPY requirements.txt /usr/local/airflow/requirements.txt

RUN pip install --no-cache-dir -r /usr/local/airflow/requirements.txt
