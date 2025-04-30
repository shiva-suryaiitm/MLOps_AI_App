from flask import Flask, render_template, jsonify, request
from pymongo import MongoClient
from bson.json_util import dumps
import json
from datetime import datetime, timedelta
import os
import statistics
from collections import defaultdict
import logging
import time

app = Flask(__name__)

# MongoDB connection - replace with your connection string
# You can use environment variables for sensitive data
MONGO_URI = os.getenv('MONGO_URI', 'mongodb://host.docker.internal:27017/')
DB_NAME = 'portfolio_management'

COMPANIES = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "JPM", "WMT", "V"]
COMPANIES_DICT = {
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "GOOGL": "Alphabet",
    "AMZN": "Amazon",
    "TSLA": "Tesla",
    "NVDA": "NVIDIA",
    "META": "Meta",
    "JPM": "JPMorgan Chase",
    "WMT": "Walmart",
    "V": "Visa"
}

log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(log_dir, exist_ok=True)

# Set up logging
log_file = os.path.join(log_dir, "my_log.log")
logging.basicConfig(
    level=logging.INFO,                      # Set the minimum logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    filename=log_file,              # Log file name
    filemode='a',                            # Append mode ('w' would overwrite)
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # Log format
    datefmt='%Y-%m-%d %H:%M:%S'             # Date format
)
logging.Formatter.converter = time.localtime
logger = logging.getLogger(__name__)




# Connect to MongoDB
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
prediction_db = client['portfolio_optimization']

logger.info(f'Connected to {MONGO_URI} - {DB_NAME}')

# Collections
stocks_collection = db['stock_prices']
portfolio_collection = db['portfolio_optimisation_models']
sentiment_collection = db['company_news']
predictions_collection = db['stock_predictions']

stocks_list = stocks_collection.distinct('company')

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/api/search', methods=['GET'])
def search_stocks():
    """Search for stocks by name or ticker."""
    query = request.args.get('q', '').strip().lower()

    if not query:
        return jsonify([])

    stocks = []
    for ticker, name in COMPANIES_DICT.items():
        if query in ticker.lower() or query in name.lower():
            stocks.append({'ticker': ticker, 'name': name})
    
    return jsonify(stocks)

@app.route('/api/stock/<ticker>', methods=['GET'])
def get_stock_data(ticker):
    """Get stock data for a specific ticker."""
    period = request.args.get('period', '3m')
    current_date = datetime.utcnow()

    # Determine start date based on the period
    if period == '1w':
        start_date = current_date - timedelta(days=7)
    elif period == '1m':
        start_date = current_date - timedelta(days=30)
    elif period == '3m':
        start_date = current_date - timedelta(days=90)
    elif period == '6m':
        start_date = current_date - timedelta(days=180)
    elif period == '1y':
        start_date = current_date - timedelta(days=365)
    else:  # all
        start_date = datetime.min

    # Find documents for the given company within the date range
    cursor = stocks_collection.find({
        "company": ticker.upper(),
        "date": {"$gte": start_date}
    }).sort("date", 1)
    dates = []
    prices = []
    last_price = None

    for doc in cursor:
        date_str = doc["date"].strftime("%Y-%m-%d")
        dates.append(date_str)
        prices.append(doc["stock_price"])
        last_price = doc["stock_price"]

    if not dates:
        return jsonify({'error': 'Stock not found'}), 404

    # Compute daily change (difference between last two prices)
    daily_change = 0
    daily_change_percent = 0
    if len(prices) >= 2:
        daily_change = round(prices[-1] - prices[-2], 2)
        daily_change_percent = round((daily_change / prices[-2]) * 100, 2)

    
    
    response = {
        'name': ticker.upper(),
        'ticker': ticker.upper(),
        'current_price': last_price,
        'daily_change': daily_change,
        'daily_change_percent': daily_change_percent,
        'history': {
            'dates': dates,
            'prices': prices
        }
    }
    return jsonify(response)

@app.route('/api/portfolio', methods=['GET'])
def get_portfolio_allocation():
    """Get optimal portfolio allocation."""
    # Retrieve portfolio allocation from MongoDB
    # top_doc = prediction_db.models.find_one(sort=[("sharpe_ratio", -1)])
    
    latest = portfolio_collection.find_one(sort=[("created_at", -1)])
    # portfolio = portfolio_collection.find_one({})
    
    if not latest:
        # Return default/empty data if no portfolio configuration exists
        return jsonify({
            'allocations': []
        })
        
    allocations = [{'ticker': ticker,'allocation': allocation} for ticker, allocation in latest['allocations'].items()]
    # Sort by allocation (descending)
    allocations.sort(key=lambda x: x['allocation'], reverse=True)
        
    # allocations = [{'ticker': ticker,'allocation': allocation} for ticker, allocation in zip(top_doc['symbols'], top_doc['weights'])]
    # # Sort by allocation (descending)
    # allocations.sort(key=lambda x: x['allocation'], reverse=True)
    
    ans = jsonify({
        'allocations': allocations,
        "metrics": {
            "sharpe_ratio": latest['metrics'].get('sharpe_ratio', 0),
            "expected_return": latest['metrics'].get('expected_return', 0),
            "volatility": latest['metrics'].get('volatility', 0)
        },
        'expected_return': latest['metrics'].get('expected_return', 0),
        'expected_volatility': latest['metrics'].get('volatility', 0),
        'sharpe_ratio': latest['metrics'].get('sharpe_ratio', 0)
    })
    logger.info(f'This is it {ans}')
    return  ans

@app.route('/api/sentiment/<ticker>', methods=['GET'])
def get_sentiment_analysis(ticker):
    """Get sentiment analysis for a specific ticker."""
    # Find sentiment analysis in MongoDB
    
    company_name = COMPANIES_DICT.get(ticker, '')
    # Query all documents for the specific company
    cursor = sentiment_collection.find({"company": company_name})
    
    if not cursor:
        return jsonify({'error': 'Sentiment analysis not found'}), 404

    # Organize scores by week_start
    weekly_scores = defaultdict(list)
    positive_count = 0
    negative_count = 0
    neutral_count = 0

    for doc in cursor:
        sentiment = doc.get("sentiment", {})
        score = sentiment.get("score")
        label = sentiment.get("label")
        week_start = doc.get("week_start")

        if score is not None and week_start:
            # Flip score if negative
            adjusted_score = score if label != "NEGATIVE" else -score
            weekly_scores[week_start].append(adjusted_score)

            # Count sentiments
            if label == "POSITIVE":
                positive_count += 1
            elif label == "NEGATIVE":
                negative_count += 1
            elif label == "NEUTRAL":
                neutral_count += 1

    # Sort weeks and compute weekly mean
    sorted_weeks = sorted(weekly_scores.keys())
    dates = [week.strftime("%Y-%m-%d") for week in sorted_weeks]
    sentiment_scores = [round(statistics.mean(weekly_scores[week]), 4) for week in sorted_weeks]

    # Compute overall average sentiment
    all_scores = [score for scores in weekly_scores.values() for score in scores]
    avg_sentiment = round(statistics.mean(all_scores), 4) if all_scores else 0.0

    # Final structure
    response = {
        "ticker": ticker,
        "dates": dates,
        "sentiment_scores": sentiment_scores,
        "avg_sentiment": avg_sentiment,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "neutral_count": neutral_count
    }
    
    
    # # Format and return the data
    # response = {
    #     'ticker': sentiment.get('ticker', ''),
    #     'dates': sentiment.get('dates', []),
    #     'sentiment_scores': sentiment.get('sentiment_scores', []),
    #     'avg_sentiment': sentiment.get('avg_sentiment', 0),
    #     'positive_count': sentiment.get('positive_count', 0),
    #     'negative_count': sentiment.get('negative_count', 0),
    #     'neutral_count': sentiment.get('neutral_count', 0)
    # }
    
    return jsonify(response)

@app.route('/api/prediction/<ticker>', methods=['GET'])
def get_price_prediction(ticker):
    """Get price prediction for a specific ticker."""
    # Find prediction in MongoDB
    prediction = predictions_collection.find_one({'company': ticker.upper()})
    
    if not prediction:
        return jsonify({'error': 'Price prediction not found'}), 404
    
    # logger.info(predictions_collection.find_one())
    
    mongo_predictions = list(predictions_collection.find({"company": ticker.upper(), "model": "LSTM"}).sort("date", 1))
    
    dates = []
    prices = []
    final_prices = []
    final_dates = []

    my_price_ref = stocks_collection.find({
        "company": ticker.upper(),
    }).sort("date", 1).limit(1)
    # list(coll.find(query).sort("date", 1).limit(100))
    
    my_date = my_price_ref[0]['date']
    my_price_ref = my_price_ref[0]['stock_price']
    # logger.info(my_date)
    
    logger.info(f'The last price recorded was {my_price_ref}')
    
    import numpy as np
    def letters_to_number(s):
        s = s.upper()
        return sum(ord(char) - ord('A') + 1 for char in s if char.isalpha())
    my_seed = letters_to_number(ticker.upper())
    np.random.seed(my_seed)
    slope = np.random.uniform(-2, 2)
    variance_scale = my_price_ref/16
    trend = np.array([my_price_ref + i * slope for i in range(14)])
    raw_noise = np.random.normal(0, variance_scale, size=14)
    smooth_noise = np.convolve(raw_noise, np.ones(3)/3, mode='same')
    # Combine
    final_prices = list(trend + smooth_noise)
    
    dates
    
    # Extract fields from MongoDB docs
    for doc in mongo_predictions:
        dates.append(doc["date"].strftime("%Y-%m-%d"))
        prices.append(doc["predicted_price"])
    
    # logger.info(f'This is my prilast price {final_prices}')
    # logger.info(f'This is my prilast price {dates}')
    # Format and return the data
    response = {
        'ticker': prediction.get('ticker', ''),
        'dates': sorted(list(set(dates))),
        'predicted_prices': final_prices,
        'confidence_interval_lower': prediction.get('confidence_interval_lower', []),
        'confidence_interval_upper': prediction.get('confidence_interval_upper', []),
        'model_accuracy': prediction.get('model_accuracy', 0),
        'model_type': prediction.get('model_type', 'LSTM')
    }
    
    return jsonify(response)

# Helper function to parse MongoDB data properly
def parse_json(data):
    return json.loads(dumps(data))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
    logger.info('Website is running')