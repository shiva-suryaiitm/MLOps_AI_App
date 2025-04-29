import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.dirname(__file__))) )

import logging
from airflow.decorators import task
from utils.mongodb_client import MongoDBClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@task
def store_news(news_data):
    """
    Store news articles in MongoDB for each company.
    
    Args:
        news_data: List of tuples, each containing (company, articles, week_start, week_end)
                   - company: str, company name
                   - articles: list, news articles from NewsAPI
                   - week_start: datetime, start of the week
                   - week_end: datetime, end of the week
    
    Returns:
        None
    """
    mongo_client = MongoDBClient()
    
    try:
        for company, articles, week_start, week_end in news_data:
            if articles:
                logger.info(f"Storing {len(articles)} articles for {company}")
                mongo_client.store_news(company, articles, week_start, week_end)
            else:
                logger.warning(f"No articles to store for {company}")
    except Exception as e:
        logger.error(f"Error storing news articles: {str(e)}")
        raise
    finally:
        mongo_client.client.close()
        logger.info("MongoDB connection closed")