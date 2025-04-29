from flask import Flask, render_template, jsonify, request
from pymongo import MongoClient
from bson.json_util import dumps
import json
from datetime import datetime, timedelta
import os
import statistics
from collections import defaultdict

app = Flask(__name__)

# MongoDB connection - replace with your connection string
# You can use environment variables for sensitive data
MONGO_URI = os.environ.get('MONGO_URI', 'mongodb://localhost:27017/')
DB_NAME = os.environ.get('DB_NAME', 'portfolio_management')

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


# Connect to MongoDB
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
prediction_db = client['portfolio_optimization']

# Collections
stocks_collection = db['stock_prices']
portfolio_collection = db['portfolio']
sentiment_collection = db['company_news']
predictions_collection = db['predictions']

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
    period = request.args.get('period', '6m')
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
    top_doc = prediction_db.models.find_one(sort=[("sharpe_ratio", -1)])
    # portfolio = portfolio_collection.find_one({})
    
    if not top_doc:
        # Return default/empty data if no portfolio configuration exists
        return jsonify({
            'allocations': []
        })
    
    # # Format and return the data
    # allocations = []
    
    # for ticker, allocation in portfolio.get('allocations', {}).items():
    #     allocations.append({
    #         'ticker': ticker,
    #         'allocation': allocation
    #     })
    allocations = [{'ticker': ticker,'allocation': allocation} for ticker, allocation in zip(top_doc['symbols'], top_doc['weights'])]
    # Sort by allocation (descending)
    allocations.sort(key=lambda x: x['allocation'], reverse=True)
    
    return jsonify({
        'allocations': allocations,
        'expected_return': top_doc.get('portfolio_return', 0),
        'expected_volatility': top_doc.get('portfolio_volatility', 0),
        'sharpe_ratio': top_doc.get('sharpe_ratio', 0)
    })

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
    
    print(response)
    
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
    prediction = predictions_collection.find_one({'ticker': ticker.upper()})
    
    if not prediction:
        return jsonify({'error': 'Price prediction not found'}), 404
    
    # Format and return the data
    response = {
        'ticker': prediction.get('ticker', ''),
        'dates': prediction.get('future_dates', []),
        'predicted_prices': prediction.get('predicted_prices', []),
        'confidence_interval_lower': prediction.get('confidence_interval_lower', []),
        'confidence_interval_upper': prediction.get('confidence_interval_upper', []),
        'model_accuracy': prediction.get('model_accuracy', 0),
        'model_type': prediction.get('model_type', '')
    }
    
    return jsonify(response)

# Helper function to parse MongoDB data properly
def parse_json(data):
    return json.loads(dumps(data))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)