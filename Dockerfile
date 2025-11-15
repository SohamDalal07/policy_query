# Multi-stage build to keep runtime image lean
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System deps required by PyMuPDF, FAISS, etc.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "hackrx_faiss_api:app", "--host", "0.0.0.0", "--port", "8000"]
