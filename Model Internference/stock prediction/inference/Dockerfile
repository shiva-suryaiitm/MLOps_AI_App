FROM python:3.12

# Set working directory

WORKDIR /app


RUN apt-get update && apt-get install -y --no-install-recommends gcc  python3-dev && apt-get clean && rm -rf /var/lib/apt/lists/*

# Upgrade pip

RUN python -m pip install --upgrade pip
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY stock_prediction.py . 
COPY models/ ./models/
COPY inference_config.py .

ENV PYTHONUNBUFFERED=1

# Command to run the script

CMD ["python", "stock_prediction.py"]