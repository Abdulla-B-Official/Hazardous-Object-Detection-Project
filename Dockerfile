# Use lightweight base Python image
FROM python:3.10-slim

# Prevent Ultralytics from raising read-only directory warnings in Docker
ENV YOLO_CONFIG_DIR=/tmp

# Install system dependencies needed for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for faster Docker caching
COPY requirements.txt .

# Install PyTorch CPU first to avoid heavy GPU downloads, then rest of requirements
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Expose port and run Gunicorn with a 120s timeout to prevent 502 Bad Gateway errors
EXPOSE 5000
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--threads", "2", "--timeout", "120", "app:app"]