from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
import sys
import os
sys.path.insert(0,os.path.abspath(os.path.dirname(__file__)))

from tasks.fetch_news import fetch_weekly_news
from tasks.store_news import store_news as store_weekly_news
from tasks.check_historical_news import check_historical_news
from tasks.backfill_historical_news import backfill_historical_news
from utils.config import DAG_DEFAULT_ARGS


def check_historical_news_data(ti):
    from tasks.check_historical_news import check_historical_news
    """
    Check if historical news exists for the past year for each company.
    
    Returns:
        list: List of (company, week_start) tuples for missing weeks
    """
    missing_weeks = check_historical_news()
    print(f"Missing weeks: {missing_weeks}")
    ti.xcom_push(key="missing_weeks", value=missing_weeks)

    return missing_weeks


def backfill_historical_news_data(ti):
    from tasks.backfill_historical_news import backfill_historical_news
    """
    Backfill historical news for the past year for each company.
    
    Returns:
        None
    """
    missing_weeks = ti.xcom_pull(task_ids='checking_the_historical_news_data', key='missing_weeks')
    if missing_weeks:
        backfill_historical_news(missing_weeks)
    else:
        print("No missing weeks to backfill.")

    return None

def fetch_weekly_news_data(ti):
    from tasks.fetch_news import fetch_weekly_news
    news_results = fetch_weekly_news()
    ti.xcom_push(key="news_results", value=news_results)
    return None

def store_weekly_news_data(ti):
    from tasks.store_news import store_news
    news_results = ti.xcom_pull(task_ids='fetching_the_weekly_news', key='news_results')
    if news_results:
        store_news(news_results)
    else:
        print("No news results to store.")

    return None


# Define the DAG
with DAG(
    dag_id="portfolio_news_pipeline",
    default_args=DAG_DEFAULT_ARGS,
    description="Pipeline to fetch and store company news weekly",
    schedule_interval=timedelta(weeks=1),
    start_date=datetime(2025, 4, 28),
    catchup=False,
) as dag:
    
    # # Tasks
    # check_historical = PythonOperator(
    #     task_id = 'checking_the_historical_news_data',
    #     python_callable = check_historical_news_data,
    #     provide_context = True,  
    # )

    # backfill_historical = PythonOperator(
    #     task_id = 'backfilling_the_historical_news_data',
    #     python_callable = backfill_historical_news_data,
    #     provide_context = True,  
    # )
    # # Fetch and store news tasks

    # fetch_news = PythonOperator(
    #     task_id = 'fetching_the_weekly_news',
    #     python_callable = fetch_weekly_news_data,
    #     provide_context = True,  
    # )

    # store_news = PythonOperator(
    #     task_id = 'storing_the_weekly_news',
    #     python_callable = store_weekly_news_data,
    #     provide_context = True,  
    # )

    # Tasks
    check_historical = check_historical_news()
    backfill_historical = backfill_historical_news(check_historical)
    fetch_news = fetch_weekly_news()
    store_news = store_weekly_news(fetch_news)
    
    # Dependencies
    check_historical >> backfill_historical >> fetch_news >> store_news
    
    # Dependencies
    check_historical >> backfill_historical >> fetch_news >> store_news