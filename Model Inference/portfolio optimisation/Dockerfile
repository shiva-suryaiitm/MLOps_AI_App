FROM python:3.9

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    python3-distutils \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY predict.py .
COPY ["Model Backup/", "./Model Backup/"]

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Command to run the script
CMD ["python", "predict.py"] 