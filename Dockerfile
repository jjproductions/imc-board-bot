# Dockerfile
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=on \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100

WORKDIR /app

# Install system dependencies
# cv2 (opencv-python) requires libgl1 and libglib2.0-0
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
# Remove macOS-specific dependencies (ocrmac, pyobjc) which fail on Linux
RUN grep -vE "ocrmac|pyobjc" requirements.txt > requirements.linux.txt
RUN pip install --no-cache-dir -r requirements.linux.txt

# Copy application code
COPY app ./app
COPY scripts ./scripts

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
