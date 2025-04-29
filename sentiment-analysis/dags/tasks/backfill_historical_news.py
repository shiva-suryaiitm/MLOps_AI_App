import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.dirname(__file__))) )

import logging
from datetime import timedelta
from utils.news_api_client import NewsClient as NewsApiClient
from utils.mongodb_client import MongoDBClient
from airflow.decorators import task

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@task
def backfill_historical_news(missing_weeks):
    """
    Backfill missing historical news for companies.
    
    Args:
        missing_weeks (list): List of (company, week_start) tuples
    """
    news_client = NewsApiClient()
    mongo_client = MongoDBClient()
    
    for company, week_start in missing_weeks:
        week_end = week_start + timedelta(days=7)
        logger.info(f"Backfilling news for {company} for week {week_start}")
        articles = news_client.fetch_news(company, week_start, week_end)

        if articles:
            print(f"Fetched {len(articles)} articles for {company} for week {week_start}")
            # Store the articles in MongoDB
            mongo_client.store_news(company, articles, week_start, week_end)
            print(f"Stored {len(articles)} articles for {company} for week {week_start}")
            
        else:
            logger.warning(f"No articles found for {company} for week {week_start}")