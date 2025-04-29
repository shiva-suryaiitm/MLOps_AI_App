# Portfolio Optimization Model Inference

This directory contains code for performing inference with the optimal portfolio allocation model.

## Features

- Load trained model from MongoDB or local file
- Calculate portfolio performance metrics
- Generate investment recommendations
- Visualize portfolio performance
- REST API for integration

## Requirements

Install the required packages:

```
pip install -r requirements.txt
```

## Environment Setup

Create a `.env` file with the following variables:

```
MONGO_URI=mongodb://username:password@host:port/database
```

## Usage

### Command Line Interface

Run the prediction script to generate a recommendation:

```
python predict.py
```

### API

Start the REST API server:

```
python api.py
```

The API will be available at `http://localhost:8000` with the following endpoints:

- `GET /` - API information
- `GET /allocation` - Get the optimal portfolio allocation
- `GET /performance?days=30` - Calculate portfolio performance metrics
- `GET /recommendation?days=30` - Generate investment recommendation
- `GET /chart?days=30` - Get portfolio performance chart

### API Documentation

Once the server is running, interactive API documentation is available at:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Model Source

The model inference component attempts to load the portfolio optimization model in the following order:

1. From MongoDB (if connection details are provided)
2. From local file in the Model Training directory
3. Fallback to a default equal-weight portfolio if no model is found
