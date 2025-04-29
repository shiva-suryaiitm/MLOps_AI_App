import os
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__))) 

import logging
from pymongo import MongoClient
from datetime import datetime
from config import MONGO_URI, MONGO_DB_NAME, MONGO_COLLECTION_NAME

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MongoDB configuration
# MONGO_URI = "mongodb://localhost:27017/"  # Local MongoDB instance
# MONGO_DB_NAME = "portfolio_management"
# MONGO_COLLECTION_NAME = "company_news"

class MongoDBClient:
    def __init__(self):
        """Initialize MongoDB client."""
        try:
            self.client = MongoClient(MONGO_URI, timeoutMS=1000)
            self.db = self.client[MONGO_DB_NAME]
            self.collection = self.db[MONGO_COLLECTION_NAME]
            logger.info("Connected to MongoDB successfully")
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {str(e)}")
            raise

    def store_news(self, company, articles, week_start, week_end):
        """
        Store news articles in MongoDB.
        
        Args:
            company (str): Company name
            articles (list): List of news articles
            week_start (datetime): Start of the week
            week_end (datetime): End of the week
        """
        try:
            for article in articles:
                document = {
                    "company": company,
                    "headline": article.get("title", ""),
                    "source": article.get("source", {}).get("name", ""),
                    "url": article.get("url", ""),
                    "published_at": article.get("publishedAt", ""),
                    "week_start": week_start,
                    "week_end": week_end,
                    "created_at": datetime.utcnow(),
                    "content": article.get("content", article.get("description", ""))
                }
                self.collection.insert_one(document)
            logger.info(f"Stored {len(articles)} articles for {company} for week {week_start}")
        except Exception as e:
            logger.error(f"Error storing news for {company}: {str(e)}")
            raise

    def check_historical_news(self, company, week_start):
        """
        Check if news exists for a company for a specific week.
        
        Args:
            company (str): Company name
            week_start (datetime): Start of the week
        
        Returns:
            bool: True if news exists, False otherwise
        """
        try:
            count = self.collection.count_documents({
                "company": company,
                "week_start": week_start
            })
            exists = count > 0
            logger.info(f"News for {company} on week {week_start}: {'exists' if exists else 'missing'}")
            return exists
        except Exception as e:
            logger.error(f"Error checking historical news for {company}: {str(e)}")
            return False

    def update_sentiment(self, company, url, week_start, sentiment):
        """Update an article's document with sentiment analysis results."""
        try:
            result = self.collection.update_one(
                {
                    "company": company,
                    "url": url,
                    "week_start": week_start
                },
                {
                    "$set": {"sentiment": sentiment}
                }
            )
            if result.matched_count == 0:
                logger.warning(f"No article found to update sentiment for {company}, URL: {url}, week_start: {week_start}")
            else:
                logger.info(f"Updated sentiment for article: {company}, URL: {url}")
        except Exception as e:
            logger.error(f"Error updating sentiment for {company}, URL: {url}: {str(e)}")
            raise

MongoDBClient()