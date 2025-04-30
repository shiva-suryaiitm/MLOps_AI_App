import os
from datetime import timedelta

# NewsAPI configuration
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "ee40c83e071e4f4cba2dd6b566b4d4eb")  # Replace with your NewsAPI key
NEWS_API_PAGE_SIZE = 10  # Number of articles to fetch per company per week

# MongoDB configuration
MONGO_URI = "mongodb://host.docker.internal:27017/"  # Local MongoDB instance
MONGO_DB_NAME = "portfolio_management"
MONGO_COLLECTION_NAME = "company_news"

# Airflow DAG configuration
DAG_DEFAULT_ARGS = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(seconds=10),
}

# List of companies to track
COMPANIES = ["Apple", "Microsoft", "Alphabet", "Amazon", "Tesla", "NVIDIA", "Meta", "JPMorgan", "Walmart", "Visa"]
  # Add your companies here

# Historical backfill settings
HISTORICAL_WEEKS = 12  # Number of weeks to check/backfill (1 year)