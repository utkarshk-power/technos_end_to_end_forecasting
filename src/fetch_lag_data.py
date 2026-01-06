# We will be reading using lag data of previous 5 time steps from our database

import influxdb_client
from influxdb_client.client.write_api import SYNCHRONOUS
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import yaml

# Load YAML parameters
with open("inference.yaml", "r") as file:
    inference_all = yaml.safe_load(file)
    inference_params = inference_all["lag_data"]

bucket_name = inference_params["bucket_name"]
api_token = inference_params["api_token"]
url = inference_params["url"]
org = inference_params["org"]
#lag_start = inference_params["lag_start_date"]
#lag_end = inference_params["lag_end_date"]
client = influxdb_client.InfluxDBClient(url=url, token=api_token, org=org)
query_api = client.query_api()
fields = inference_params["load_fields"]

fields_filter = " or ".join([f'r["_field"] == "{field}"' for field in fields])

# Let us comment out the query code section

query = f'''
from(bucket: "{bucket_name}")
  |> range(start: 2025-12-24T11:00:00Z, stop: 2025-12-24T16:00:00Z)
  |> filter(fn: (r) => 
      r["_measurement"] == "Interconnection" or
      r["_measurement"] == "PcsData" or
      r["_measurement"] == "Ess Data"
      )
  |> filter(fn: (r) => {fields_filter})
  |> aggregateWindow(every: 60m, fn: mean, createEmpty: false)
  |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
'''


site_data = query_api.query(query=query, org=org)
site_data_df = pd.DataFrame([record.values for table in site_data for record in table.records])
print(site_data_df)
site_data_df.to_csv(inference_params["output"])
