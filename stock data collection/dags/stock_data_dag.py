from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.dummy import DummyOperator
from airflow.utils.dates import days_ago
from datetime import timedelta, datetime
from pymongo import MongoClient
import requests
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
MONGO_URI = "mongodb://host.docker.internal:27017/"
MONGO_DB_NAME = "portfolio_management"
PRICE_COLLECTION = "stock_prices"
MARKETSTACK_API_KEY = "748196c80c837166fa5bea3242afff52"  # Replace with your Marketstack API key
COMPANIES = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "JPM", "WMT", "V"]
BASE_URL = "http://api.marketstack.com/v1/eod"

def check_database_empty(**kwargs):
    """Check if stock_prices collection is empty."""
    client = MongoClient(MONGO_URI)
    db = client[MONGO_DB_NAME]
    collection = db[PRICE_COLLECTION]
    is_empty = collection.count_documents({}) == 0
    client.close()
    logger.info(f"Database is {'empty' if is_empty else 'not empty'}")
    return is_empty

def fetch_historical_data(**kwargs):
    """Fetch 6 months of historical data for a company if database is empty."""
    company = kwargs['company']
    ti = kwargs['ti']
    is_empty = ti.xcom_pull(task_ids='check_database_empty')

    if not is_empty:
        logger.info(f"Skipping historical fetch for {company}: database not empty")
        return

    params = {
        "access_key": MARKETSTACK_API_KEY,
        "symbols": company,
        "date_from": (datetime.now() - timedelta(days=180)).strftime("%Y-%m-%d"),
        "date_to": datetime.now().strftime("%Y-%m-%d"),
        "limit": 180
    }

    try:
        response = requests.get(BASE_URL, params=params)
        response.raise_for_status()
        data = response.json()

        if "data" not in data or not data["data"]:
            logger.error(f"No historical data for {company}: {data.get('error', 'Unknown error')}")
            return

        client = MongoClient(MONGO_URI)
        db = client[MONGO_DB_NAME]
        collection = db[PRICE_COLLECTION]

        records = [
            {
                "company": company,
                "date": datetime.strptime(item["date"], "%Y-%m-%dT%H:%M:%S%z").replace(tzinfo=None),
                "stock_price": float(item["close"])
            }
            for item in data["data"]
        ]

        if records:
            collection.insert_many(records)
            logger.info(f"Stored {len(records)} historical records for {company}")
        else:
            logger.warning(f"No historical records to store for {company}")

        client.close()
    except Exception as e:
        logger.error(f"Error fetching historical data for {company}: {e}")

def fetch_daily_data(**kwargs):
    """Fetch today's stock price for a company."""
    company = kwargs['company']
    params = {
        "access_key": MARKETSTACK_API_KEY,
        "symbols": company,
        "date_from": datetime.now().strftime("%Y-%m-%d"),
        "date_to": datetime.now().strftime("%Y-%m-%d"),
        "limit": 1
    }

    try:
        response = requests.get(BASE_URL, params=params)
        response.raise_for_status()
        data = response.json()

        if "data" not in data or not data["data"]:
            logger.error(f"No daily data for {company}: {data.get('error', 'Unknown error')}")
            return

        client = MongoClient(MONGO_URI)
        db = client[MONGO_DB_NAME]
        collection = db[PRICE_COLLECTION]

        item = data["data"][0]
        record = {
            "company": company,
            "date": datetime.strptime(item["date"], "%Y-%m-%dT%H:%M:%S%z").replace(tzinfo=None),
            "stock_price": float(item["close"])
        }

        # Check for duplicate
        if not collection.find_one({"company": company, "date": record["date"]}):
            collection.insert_one(record)
            logger.info(f"Stored daily price for {company} on {record['date']}")
        else:
            logger.info(f"Daily price for {company} on {record['date']} already exists")

        client.close()
    except Exception as e:
        logger.error(f"Error fetching daily data for {company}: {e}")

def create_dag():
    default_args = {
        'owner': 'airflow',
        'depends_on_past': False,
        'email_on_failure': False,
        'email_on_retry': False,
        'retries': 2,
        'retry_delay': timedelta(seconds=10),
    }

    dag = DAG(
        'stock_data_collection',
        default_args=default_args,
        description='Fetch historical and daily stock data for 10 companies',
        schedule_interval='0 18 * * *',  # Run daily at 6 PM EST (11 PM IST)
        start_date=days_ago(1),
        catchup=False,
        max_active_runs=1,
    )

    start = DummyOperator(task_id='start', dag=dag)
    check_db = PythonOperator(
        task_id='check_database_empty',
        python_callable=check_database_empty,
        dag=dag,
    )

    # Historical and daily tasks for each company
    for company in COMPANIES:
        historical_task = PythonOperator(
            task_id=f'fetch_historical_{company}',
            python_callable=fetch_historical_data,
            op_kwargs={'company': company},
            dag=dag,
        )
        daily_task = PythonOperator(
            task_id=f'fetch_daily_{company}',
            python_callable=fetch_daily_data,
            op_kwargs={'company': company},
            dag=dag,
        )

        # Chain: start -> check_db -> historical -> daily
        start >> check_db >> historical_task >> daily_task

    return dag

dag = create_dag()