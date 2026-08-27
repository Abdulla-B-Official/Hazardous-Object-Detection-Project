# Use lightweight base Python image
FROM python:3.10-slim

# Prevent Ultralytics from raising read-only directory warnings in Docker
ENV YOLO_CONFIG_DIR=/tmp

# Install system dependencies needed for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements file first
COPY requirements.txt .

# Install PyTorch CPU and remaining requirements with extra index URL in one command
RUN pip install --no-cache-dir \
    torch torchvision \
    --extra-index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port and start Gunicorn worker
EXPOSE 5000
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--threads", "2", "--timeout", "120", "app:app"]