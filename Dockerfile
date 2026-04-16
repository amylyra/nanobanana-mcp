FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir google-cloud-storage>=2.0.0 boto3>=1.34.0

COPY server.py .

ENV PORT=8080

CMD ["python", "server.py"]
