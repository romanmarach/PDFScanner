FROM python:3.12-slim

# Install system dependencies for PaddleOCR
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY agent/ ./agent/

# Create output directory
RUN mkdir -p /app/output

# Default command
ENTRYPOINT ["python", "-m", "agent"]
