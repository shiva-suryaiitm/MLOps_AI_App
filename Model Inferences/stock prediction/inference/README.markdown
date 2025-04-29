# Stock Price and Volume Prediction Inference

## Overview
This folder provides a dockerized service to predict stock prices (LSTM) and volumes (GRU) for the next 14 days, fetching 100 days from MongoDB. It uses pre-trained model weights and is GPU-accelerated.

## Prerequisites
- Docker and Docker Compose
- NVIDIA GPU with drivers (e.g., version 535+ for CUDA 12.1)
- NVIDIA Container Toolkit
- MongoDB with `stock_prices` and `stock_volumes`
- Model weights (`models/price_model.pth`, `models/volume_model.pth`)
- 4GB GPU memory

## Setup
1. **Copy model weights**:
   ```bash
   cp ../training/models/price_model.pth models/
   cp ../training/models/volume_model.pth models/
   ```

2. **Verify NVIDIA GPU**:
   ```bash
   nvidia-smi
   ```

3. **Install NVIDIA Container Toolkit**:
   ```bash
   sudo apt-get update
   sudo apt-get install -y nvidia-container-toolkit
   sudo nvidia-ctk runtime configure --runtime=docker
   sudo systemctl restart docker
   ```
   Verify:
   ```bash
   docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu20.04 nvidia-smi
   ```

4. **Start Docker Services**:
   ```bash
   docker-compose up -d
   ```

5. **Check Predictions**:
   ```bash
   docker-compose logs stock-prediction
   ```
   Expect: `Saved 14 predictions for Apple (LSTM/GRU)`.

## Configuration
- **MongoDB**: `mongodb://mongo:27017/`. Update `stock_prediction.py` if different.
- **Company**: Defaults to “Apple”. Modify `docker-compose.yaml` or run:
  ```bash
  docker run --rm --gpus all -v $(pwd)/models:/app/models stock-prediction:latest python3.8 stock_prediction.py --company "Microsoft"
  ```
- **Weights**: Stored in `models/`.

## Project Structure
```
inference/
├── stock_prediction.py       # Prediction
├── Dockerfile                # Inference service
├── requirements.txt          # Dependencies
├── docker-compose.yaml       # Docker services
├── models/                   # Model weights
└── README.md                 # Documentation
```

## Usage
- **Prediction**:
  - Fetches 100 days, predicts 14 days, saves to `stock_predictions`, `volume_predictions`.
  - Run manually:
    ```bash
    docker run --rm --gpus all -v $(pwd)/models:/app/models stock-prediction:latest python3.8 stock_prediction.py --company "Apple"
    ```

## Troubleshooting
- **GPU**: Verify `nvidia-smi`, `docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu20.04 nvidia-smi`.
- **MongoDB**: Check `docker-compose logs mongo`.
- **Weights**: Verify `ls -l models/`.

## License
Apache License 2.0