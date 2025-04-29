import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.dirname(__file__))) )

from airflow.decorators import task
import logging
from datetime import datetime, timedelta
from utils.news_api_client import NewsClient as NewsApiClient
from utils.config import COMPANIES

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@task
def fetch_weekly_news():
    """
    Fetch top 10 news articles for each company for the current week.
    
    Returns:
        list: List of (company, articles, week_start, week_end) tuples
    """
    news_client = NewsApiClient()
    current_date = datetime.utcnow()
    week_end = current_date
    week_start = current_date - timedelta(days=7)
    
    results = []
    for company in COMPANIES:
        logger.info(f"Fetching news for {company}")
        articles = news_client.fetch_news(company, week_start, week_end)
        results.append((company, articles, week_start, week_end))
    
    if not results:
        logger.warning("No news fetched for any company")
    
    return results