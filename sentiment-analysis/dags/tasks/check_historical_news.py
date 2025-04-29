import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.dirname(__file__))) )


import logging
from datetime import datetime, timedelta
from utils.mongodb_client import MongoDBClient
from utils.config import COMPANIES, HISTORICAL_WEEKS
from airflow.decorators import task


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@task
def check_historical_news():
    """
    Check if historical news exists for the past year for each company.
    
    Returns:
        list: List of (company, week_start) tuples for missing weeks
    """
    mongo_client = MongoDBClient()
    print(mongo_client.collection.count_documents({}))
    missing_weeks = []
    
    for company in COMPANIES:
        for week in range(HISTORICAL_WEEKS):
            week_start = (datetime.utcnow() - timedelta(weeks=week)).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            if not mongo_client.check_historical_news(company, week_start):
                logger.info(f"Missing news for {company} for week {week_start}")
                missing_weeks.append((company, week_start))
    
    return missing_weeks