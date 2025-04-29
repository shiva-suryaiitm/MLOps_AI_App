import os
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import logging
from datetime import datetime, timedelta
from newsapi import NewsApiClient
from config import NEWS_API_KEY, NEWS_API_PAGE_SIZE

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NewsClient:
    def __init__(self):
        """Initialize NewsAPI client with API key."""
        try:
            self.client = NewsApiClient(api_key=NEWS_API_KEY)
            logger.info("NewsAPI client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize NewsAPI client: {str(e)}")
            raise

    def fetch_news(self, company, from_date, to_date):
        """
        Fetch top news for a company within a date range.
        
        Args:
            company (str): Company name
            from_date (datetime): Start date for news
            to_date (datetime): End date for news
        
        Returns:
            list: List of news articles
        """
        try:
            logger.info(f"Fetching news for {company} from {from_date} to {to_date}")
            response = self.client.get_everything(
                q=company,
                from_param=from_date.strftime("%Y-%m-%d"),
                to=to_date.strftime("%Y-%m-%d"),
                language="en",
                sort_by="relevancy",
                page_size=NEWS_API_PAGE_SIZE
            )
            articles = response.get("articles", [])
            logger.info(f"Retrieved {len(articles)} articles for {company}")
            return articles
        except Exception as e:
            logger.error(f"Error fetching news for {company}: {str(e)}")
            return []