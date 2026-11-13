FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY callbacks/ ./callbacks/
COPY datasets/ ./datasets/
COPY engine/ ./engine/
COPY models/ ./models/
COPY utils/ ./utils/
COPY train.py .
