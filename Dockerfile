FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    FLAGS_use_mkldnn=0 \
    PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True \
    FORWARDED_ALLOW_IPS=127.0.0.1,::1 \
    HOME=/home/appuser

# Install system dependencies for PaddleOCR and image/PDF processing.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && useradd --create-home --uid 1000 appuser

COPY --chown=appuser:appuser agent/ ./agent/
COPY --chown=appuser:appuser static/ ./static/
COPY --chown=appuser:appuser templates/ ./templates/
COPY --chown=appuser:appuser web_app.py ./

RUN mkdir -p /app/uploads /app/output /home/appuser/.paddlex \
    && chown -R appuser:appuser /app/uploads /app/output /home/appuser/.paddlex

USER appuser

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--threads", "4", "--timeout", "180", "--access-logfile", "-", "web_app:app"]