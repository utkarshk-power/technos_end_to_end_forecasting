FROM python:3.9-slim

WORKDIR /app

COPY requirements.infer.txt /app/requirements.infer.txt
RUN pip install --no-cache-dir -r /app/requirements.infer.txt

COPY src /app/src
COPY inference.yaml /app/inference.yaml
COPY params.yaml /app/params.yaml
COPY models /app/models
COPY docker/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

CMD ["/app/entrypoint.sh"]
