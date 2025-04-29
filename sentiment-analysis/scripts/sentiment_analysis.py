import logging
from pymongo import MongoClient
from transformers import pipeline
import torch

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# MongoDB configuration
MONGO_URI = "mongodb://host.docker.internal:27017/"
MONGO_DB_NAME = "portfolio_management"
MONGO_COLLECTION_NAME = "company_news"

def analyze_sentiment():
    """Perform sentiment analysis on articles with missing sentiment and non-empty content."""
    # Initialize MongoDB client
    try:
        client = MongoClient(MONGO_URI)
        db = client[MONGO_DB_NAME]
        collection = db[MONGO_COLLECTION_NAME]
        client.admin.command('ping')
        logger.info("Connected to MongoDB")
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {str(e)}")
        raise

    # Initialize Hugging Face sentiment analysis pipeline (CPU)
    try:
        sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device = (0 if torch.cuda.is_available() else -1)  # Force CPU
        )
        logger.info("Initialized sentiment analysis pipeline")
    except Exception as e:
        logger.error(f"Failed to initialize sentiment pipeline: {str(e)}")
        raise

    try:
        # Query articles where sentiment is missing and content is not empty
        query = {
            "sentiment": {"$exists": False},
            "content": {"$ne": ""}
        }
        articles = collection.find(query)
        article_count = collection.count_documents(query)
        logger.info(f"Found {article_count} articles to analyze")

        if article_count == 0:
            logger.info("No articles require sentiment analysis")
            return

        for article in articles:
            content = article.get("content", "")
            headline = article.get("headline", "Unknown")

            # Perform sentiment analysis
            try:
                result = sentiment_pipeline(content)[0]
                sentiment = {
                    "label": result["label"],  # e.g., "POSITIVE" or "NEGATIVE"
                    "score": result["score"]   # Confidence score (0 to 1)
                }
            except Exception as e:
                logger.error(f"Error analyzing sentiment for article '{headline}': {str(e)}")
                continue

            # Update MongoDB document
            try:
                collection.update_one(
                    {"_id": article["_id"]},
                    {"$set": {"sentiment": sentiment}}
                )
                logger.info(f"Updated sentiment {sentiment} for article: {headline}")
            except Exception as e:
                logger.error(f"Error updating sentiment for article '{headline}': {str(e)}")
                continue

    except Exception as e:
        logger.error(f"Error processing articles: {str(e)}")
        raise
    finally:
        client.close()
        logger.info("MongoDB connection closed")

if __name__ == "__main__":
    analyze_sentiment()