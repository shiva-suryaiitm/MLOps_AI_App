import os
import numpy as np
import pandas as pd
import joblib
from datetime import datetime, timedelta
import pymongo
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import json
from pymongo import MongoClient
import logging
import time
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

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

MONGO_URI = "mongodb://localhost:27017/"
MONGO_DB_NAME = "portfolio_management"
SEQUENCE_LENGTH = 100
COMPANIES = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "JPM", "WMT", "V"]


# Load environment variables
load_dotenv()
class PortfolioOptimizer:
    def __init__(self):
        """Initialize the portfolio optimizer."""
        self.model = None
        self.symbols = None
        self.weights = None
        
        # Create directory for outputs
        os.makedirs('outputs', exist_ok=True)
        
        # Load model from MongoDB or local file
        self._load_model()
    
    def _load_model(self):
        """Load the trained model either from MongoDB or local file."""
        try:
            # First try to load from MongoDB
            mongo_uri = os.getenv('MONGO_URI')
            if mongo_uri:
                client = pymongo.MongoClient(mongo_uri)
                db = client.portfolio_optimization
                collection = db.models
                
                # Get the latest model
                latest_model = collection.find_one(sort=[('training_date', pymongo.DESCENDING)])
                
                if latest_model:
                    self.symbols = latest_model['symbols']
                    self.weights = np.array(latest_model['weights'])
                    self.model = {
                        'weights': self.weights,
                        'symbols': self.symbols,
                        'training_date': latest_model['training_date'],
                        'metrics': {
                            'sharpe_ratio': latest_model.get('sharpe_ratio', 0)
                        }
                    }
                    logger.info(f"Model loaded from MongoDB, trained on {latest_model['training_date']}")
                    return
        except Exception as e:
            logger.info(f"Error loading model from MongoDB: {e}")
        
        # Fallback to local file
        try:
            model_path = './Model Backup/portfolio_model.joblib'
            if os.path.exists(model_path):
                self.model = joblib.load(model_path)
                self.symbols = self.model['symbols']
                self.weights = np.array(self.model['weights'])
                logger.info(f"Model loaded from local file, trained on {self.model['training_date']}")
            else:
                raise FileNotFoundError("Model file not found")
        except Exception as e:
            logger.info(f"Error loading model from file: {e}")
            logger.info("Using default equal-weight portfolio as fallback")
            
            # Fallback to default model with equal weights
            self.symbols = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'META', 'TSLA', 'NVDA', 'JPM', 'BAC', 'V']
            self.weights = np.ones(len(self.symbols)) / len(self.symbols)
            self.model = {
                'weights': self.weights,
                'symbols': self.symbols,
                'training_date': datetime.now().strftime('%Y-%m-%d'),
                'metrics': {
                    'sharpe_ratio': 0
                }
            }
    

    def fetch_data(self, collection, company, n_days=30):
        """Fetch data from MongoDB."""
        try:
            client = MongoClient(MONGO_URI)
            db = client[MONGO_DB_NAME]
            coll = db[collection]
            end_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            start_date = end_date - timedelta(days=n_days)
            query = {
                "company": company,
                "date": {"$gte": start_date, "$lte": end_date}
            }
            data = list(coll.find(query).sort("date", 1))
            client.close()
            if len(data) < n_days:
                logger.info(f"Only {len(data)} days of data found for {company}")
            return data
        except Exception as e:
            logger.info(f"Error fetching data: {str(e)}")
            raise

    def create_stock_price_dataframe(self, companies):
        # Initialize an empty list to store all data
        collection='stock_prices'
        all_data = []
        # Fetch data for each company
        for company in companies:
            data = self.fetch_data(collection=collection, company=company)
            for record in data:
                all_data.append({
                    'date': record['date'],
                    'company': record['company'],
                    'stock_price': record['stock_price']
                })
        
        # Create DataFrame from all data
        df = pd.DataFrame(all_data)
        
        # Pivot the DataFrame to have dates as rows and companies as columns
        pivot_df = df.pivot(index='date', columns='company', values='stock_price')
        
        # Reset index to make 'date' a column
        pivot_df = pivot_df.reset_index()
        
        # Ensure date is in datetime format
        pivot_df['date'] = pd.to_datetime(pivot_df['date'])
        
        # Sort by date
        pivot_df = pivot_df.sort_values('date')
        
        return pivot_df
    
    def get_portfolio_allocation(self):
        """Return the current portfolio allocation."""
        return {symbol: weight for symbol, weight in zip(self.symbols, self.weights)}
    
    def calculate_portfolio_performance(self, days=30):
        """Calculate performance metrics for the portfolio."""
        # Fetch recent data
        data_dict = self.create_stock_price_dataframe(companies=COMPANIES)

        # Ensure date is the index for calculations
        price_data = data_dict.set_index('date')

        # Filter for the symbols in self.symbols
        price_data = price_data[[symbol for symbol in self.symbols if symbol in price_data.columns]]

        # Drop rows with any NaN values
        price_data = price_data.dropna()

        # Calculate daily returns
        returns_data = price_data.pct_change().dropna()

        # Calculate daily portfolio returns
        # portfolio_returns = returns_data.dot(self.weights)
        # Ensure weights align with the columns in returns_data
        weights = np.array([self.weights[i] for i in range(len(returns_data.columns))])
        portfolio_returns = returns_data.dot(weights)

        # Calculate metrics
        cumulative_return = (1 + portfolio_returns).prod() - 1
        annualized_return = (1 + cumulative_return) ** (252 / len(portfolio_returns)) - 1
        volatility = portfolio_returns.std() * np.sqrt(252)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0

        # Normalize prices for visualization
        normalized_prices = price_data / price_data.iloc[0]

        # Calculate portfolio value over time
        portfolio_value = normalized_prices.dot(self.weights)

        # Create performance chart
        output_dir = 'outputs'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_path = os.path.join(output_dir, 'portfolio_performance.png')

        plt.style.use('default')
        fig, ax = plt.subplots(figsize=(14, 8))

        # Define colors for stocks
        stock_colors = plt.cm.viridis(np.linspace(0, 1, len(self.symbols)))

        # Plot portfolio
        portfolio_color = '#003366'
        ax.plot(portfolio_value.index, portfolio_value.values,
                label='Portfolio',
                color=portfolio_color,
                linewidth=3,
                zorder=10)

        # Plot individual stocks
        for i, symbol in enumerate(self.symbols):
            if symbol in normalized_prices.columns:
                ax.plot(normalized_prices.index, normalized_prices[symbol],
                        label=symbol,
                        color=stock_colors[i],
                        linewidth=1.5,
                        alpha=0.6)

        # Customize plot
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Normalized Value', fontsize=12)
        ax.set_title('Portfolio Performance Over Time', fontsize=16, fontweight='bold', pad=20)
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.,
                fontsize=10, title='Assets', title_fontsize=11, frameon=False)
        ax.grid(True, which='major', linestyle='--', linewidth=0.5, color='grey', alpha=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('grey')
        ax.spines['bottom'].set_color('grey')
        ax.tick_params(axis='x', rotation=45, colors='dimgray', labelsize=10)
        ax.tick_params(axis='y', colors='dimgray', labelsize=10)
        ax.tick_params(axis='both', direction='in', length=4)

        plt.tight_layout(rect=[0, 0, 0.85, 1])
        fig.patch.set_facecolor('none')
        ax.patch.set_facecolor('none')
        plt.savefig(output_path, dpi=300, bbox_inches='tight', transparent=True)
        logger.info(f"Plot saved to {output_path}")
        plt.close(fig)

        # Save metrics
        metrics = {
            'cumulative_return': float(cumulative_return),
            'annualized_return': float(annualized_return),
            'volatility': float(volatility),
            'sharpe_ratio': float(sharpe_ratio),
            'evaluation_date': datetime.now().strftime('%Y-%m-%d'),
            'evaluation_period_days': days
        }

        with open('outputs/performance_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=4)

        return metrics, portfolio_value
    
    def generate_recommendation(self):
        """Generate investment recommendations based on the optimal portfolio."""
        allocation = self.get_portfolio_allocation()
        metrics, _ = self.calculate_portfolio_performance()
        
        # Sort stocks by weight
        sorted_allocation = {k: v for k, v in sorted(allocation.items(), key=lambda item: item[1], reverse=True)}
        
        # Generate recommendation
        recommendation = {
            'optimal_allocation': sorted_allocation,
            'performance_metrics': metrics,
            'recommendation_date': datetime.now().strftime('%Y-%m-%d'),
            'model_training_date': str(self.model['training_date']),
        }
        
        # Save recommendation
        with open('outputs/recommendation.json', 'w') as f:
            json.dump(recommendation, f, indent=4)
        
        return recommendation


if __name__ == "__main__":
    # Create portfolio optimizer
    optimizer = PortfolioOptimizer()
    # Generate recommendation
    recommendation = optimizer.generate_recommendation()
    
    # logger.info summary
    logger.info("--------- Optimal Portfolio Allocation ---------")
    for symbol, weight in recommendation['optimal_allocation'].items():
        logger.info(f"  {symbol}: {weight:.2%}")
    
    logger.info("--------- Portfolio Performance Metrics ---------")
    metrics = recommendation['performance_metrics']
    logger.info(f"  Cumulative Return: {abs(metrics['cumulative_return']):.2%}")
    logger.info(f"  Annualized Return: {abs(metrics['annualized_return']):.2%}")
    logger.info(f"  Volatility: {metrics['volatility']:.2%}")
    logger.info(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    
    logger.info("Recommendation saved to outputs/recommendation.json")
    logger.info("Performance chart saved to outputs/portfolio_performance.png") 
    logger.info("====="*20)