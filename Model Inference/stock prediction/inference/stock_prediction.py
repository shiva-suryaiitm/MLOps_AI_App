import argparse
import logging
from datetime import datetime, timedelta
import numpy as np
from pymongo import MongoClient
import torch
import torch.nn as nn
import schedule
import time
from sklearn.preprocessing import MinMaxScaler
import os
# from inference_config import MONGO_URI, MONGO_DB_NAME, PRICE_COLLECTION, VOLUME_COLLECTION, PRICE_PREDICTION_COLLECTION, VOLUME_PREDICTION_COLLECTION
from inference_config import SEQUENCE_LENGTH, PREDICTION_DAYS

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# MongoDB configuration
MONGO_URI = "mongodb://host.docker.internal:27017/"
MONGO_DB_NAME = "portfolio_management"
PRICE_COLLECTION = "stock_prices"
VOLUME_COLLECTION = "stock_volumes"
PRICE_PREDICTION_COLLECTION = "stock_predictions"
VOLUME_PREDICTION_COLLECTION = "volume_predictions"

# # Model configuration
SEQUENCE_LENGTH = 100
PREDICTION_DAYS = 14
DEVICE = torch.device("cpu" if torch.cuda.is_available() else "cpu")

# LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# GRU Model
class GRUModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.3):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])
        return out

def fetch_data(collection, company, n_days=100):
    """Fetch data from MongoDB."""
    try:
        client = MongoClient(MONGO_URI)
        db = client[MONGO_DB_NAME]
        coll = db[collection]
        end_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        start_date = end_date - timedelta(days=n_days)
        query = {
            "company": company,
            "date": {"$lte": end_date}
        }
        data = list(coll.find(query).sort("date", 1).limit(100))
        logger.info(coll.find_one())
        client.close()
        if len(data) < n_days:
            logger.warning(f"Only {len(data)} days of data found for {company}")
        return data
    except Exception as e:
        logger.error(f"Error fetching data: {str(e)}")
        raise

def preprocess_data(data, field, sequence_length=SEQUENCE_LENGTH):
    """Preprocess data."""
    values = np.array([item[field] for item in data]).reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled_values = scaler.fit_transform(values)
    sequences = []
    for i in range(len(scaled_values) - sequence_length):
        sequences.append(scaled_values[i:i + sequence_length])
    return np.array(sequences), scaler

def predict_iterative(model, last_sequence, scaler, days=PREDICTION_DAYS):
    """Perform predictions."""
    model.eval()
    predictions = []
    current_sequence = last_sequence.copy()
    # print(current_sequence.shape)
    with torch.no_grad():
        for _ in range(days):
            input_tensor = torch.FloatTensor(current_sequence).to(DEVICE)
            # print(input_tensor.size())
            pred = model(input_tensor).cpu().numpy()
            predictions.append(pred[0, 0])
            current_sequence = np.roll(current_sequence, -1, axis=0)
            current_sequence[-1] = pred
    return scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()

def save_predictions(collection, company, predictions, model_name, start_date):
    """Save predictions."""
    try:
        client = MongoClient(MONGO_URI)
        db = client[MONGO_DB_NAME]
        coll = db[collection]
        documents = [
            {
                "company": company,
                f"predicted_{'price' if model_name == 'LSTM' else 'volume'}": float(pred),
                "date": start_date + timedelta(days=i),
                "model": model_name,
                "created_at": datetime.now()
            }
            for i, pred in enumerate(predictions)
        ]
        coll.insert_many(documents)
        client.close()
        logger.info(f"Saved {len(predictions)} predictions for {company} ({model_name})")
    except Exception as e:
        logger.error(f"Error saving predictions: {str(e)}")
        raise

def create_sequences(data, seq_length):
    """Create sequences for prediction."""
    X = []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
    return np.array(X)

def predict_all_companies():
    """Predict for all companies."""

    # Get all companies
    client = MongoClient(MONGO_URI)
    db = client[MONGO_DB_NAME]
    companies = db[PRICE_COLLECTION].distinct("company")
    client.close()
    logger.info(f"Found companies: {companies}")

    # Predict for each company
    for company in companies:
        logger.info(f"Predicting for {company}")
        try:
            predict_single_company(company, n_days = PREDICTION_DAYS)
        except Exception as e:
            logger.error(f"Error predicting for {company}: {str(e)}")

def predict_single_company(company, n_days=100):
    """Run predictions."""
    # Fetch data
    price_data = fetch_data(PRICE_COLLECTION, company, n_days)
    if len(price_data) < n_days :
        logger.error("Insufficient data for prediction")
        return

    # Predict prices
    last_price_sequence = np.array([item["stock_price"] for item in price_data[-n_days:]]).reshape(-1, 1)
    price_scaler = MinMaxScaler().fit(last_price_sequence)
    last_price_sequence = price_scaler.transform(last_price_sequence)[-SEQUENCE_LENGTH:].reshape(1, -1, 1)
    price_predictions = predict_iterative(price_model, last_price_sequence, price_scaler)
    save_predictions(PRICE_PREDICTION_COLLECTION, company, price_predictions, "LSTM", datetime.now())

    # Predict volumes
    # volume_data = fetch_data(VOLUME_COLLECTION, company, n_days)
    # last_volume_sequence = np.array([item["volume"] for item in volume_data[-n_days:]]).reshape(-1, 1)
    # volume_scaler = MinMaxScaler().fit(last_volume_sequence)
    # last_volume_sequence = volume_scaler.transform(last_volume_sequence)[-SEQUENCE_LENGTH:].reshape(1, -1, 1)
    # volume_predictions = predict_iterative(volume_model, last_volume_sequence, volume_scaler)
    # save_predictions(VOLUME_PREDICTION_COLLECTION, company, volume_predictions, "GRU", datetime.now())

def main():
    """Run predictions daily."""
    logger.info("Starting prediction scheduler")
    predict_all_companies()

    # # Schedule the prediction function to run daily at midnight
    # schedule.every().day.at("00:00").do(predict_all_companies)

    # while True:
    #     schedule.run_pending()
    #     time.sleep(60)  # Check every minute

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Stock Price and Volume Prediction")
    parser.add_argument("--company", type=str, default="AAPL", help="Company name")
    parser.add_argument("--price_weights", type=str, default= os.path.dirname(__file__) + "/models/price_model.pth", help="Path to price model weights")
    parser.add_argument("--volume_weights", type=str, default= os.path.dirname(__file__) + "/models/volume_model.pth", help="Path to volume model weights")
    args = parser.parse_args()

    # Initialize models with best hyperparameters
    price_params, price_model_weights = torch.load(args.price_weights)['params'], torch.load(args.price_weights)['weights']
    volume_params, volume_model_weights = torch.load(args.volume_weights)['params'], torch.load(args.volume_weights)['weights']

    price_model = LSTMModel(
        hidden_size=price_params["hidden_size"],
        dropout=price_params["dropout"]
    ).to(DEVICE)

    volume_model = GRUModel(
        hidden_size=volume_params["hidden_size"],
        dropout=volume_params["dropout"]
    ).to(DEVICE)

    # Load weights
    if not os.path.exists(args.price_weights):
        logger.error(f"Price model weights not found at {args.price_weights}")
    price_model.load_state_dict(price_model_weights)
    logger.info(f"Loaded price model weights from {args.price_weights}")

    if not os.path.exists(args.volume_weights):
        logger.error(f"Volume model weights not found at {args.volume_weights}")
    volume_model.load_state_dict(volume_model_weights)
    logger.info(f"Loaded volume model weights from {args.volume_weights}")

    main()