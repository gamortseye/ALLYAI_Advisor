# Dockerfile for Render: installs system deps, builds/installs python deps,
# downloads the model at container startup and runs uvicorn (FastAPI + Gradio)
FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# Install system packages required to build/ run llama-cpp-python and for audio/video features if needed
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential cmake git wget git-lfs ffmpeg libsndfile1 \
      curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python deps
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy app
COPY . /app

# Expose port (Render uses $PORT env var)
ENV PORT=8080
EXPOSE 8080

# Use gunicorn + uvicorn worker for production robustness
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "app:app", "--bind", "0.0.0.0:8080", "--workers", "1", "--timeout", "120"]
