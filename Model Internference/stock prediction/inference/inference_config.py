# MongoDB configuration
MONGO_URI = "mongodb://host.docker.internal:27017/"
MONGO_DB_NAME = "portfolio_management"
PRICE_COLLECTION = "stock_prices_fake"
VOLUME_COLLECTION = "stock_volumes_fake"
PRICE_PREDICTION_COLLECTION = "stock_predictions_fake"
VOLUME_PREDICTION_COLLECTION = "volume_predictions_fake"

# Model configuration
SEQUENCE_LENGTH = 100
PREDICTION_DAYS = 14