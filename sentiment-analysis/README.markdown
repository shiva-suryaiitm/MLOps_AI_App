# Portfolio Management News Pipeline

## Overview
This project uses Apache Airflow to fetch and store weekly news articles for portfolio companies in a MongoDB database, and a GPU-accelerated sentiment analysis service to process article content daily. The Airflow pipeline fetches news, stores it, and backfills historical news for the past year. The sentiment analysis service runs once daily, leveraging an NVIDIA GPU with PyTorch to analyze articles with missing sentiment scores.

## Prerequisites
- Docker and Docker Compose
- Python 3.8+
- NewsAPI key (sign up at https://newsapi.org/)
- NVIDIA GPU with compatible drivers (e.g., version 535+ for CUDA 12.1)
- NVIDIA Container Toolkit (`nvidia-container-toolkit`) for GPU support
- At least 8GB of disk space for Hugging Face model cache
- At least 8GB RAM and 4GB GPU memory for sentiment analysis

## Setup
1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd portfolio_management
   ```

2. **Verify NVIDIA GPU and drivers**:
   ```bash
   nvidia-smi
   ```
   Ensure your GPU is listed (e.g., `GeForce RTX 3060`).

3. **Install NVIDIA Container Toolkit** (if not installed):
   ```bash
   sudo apt-get update
   sudo apt-get install -y nvidia-container-toolkit
   sudo nvidia-ctk runtime configure --runtime=docker
   sudo systemctl restart docker
   ```
   Verify GPU access:
   ```bash
   docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu20.04 nvidia-smi
   ```

4. **Set up environment variables**:
   Create a `.env` file in the project root:
   ```env
   AIRFLOW_UID=50000
   AIRFLOW_IMAGE_NAME=custom-airflow:2.10.4
   NEWS_API_KEY=your-newsapi-key
   ```

5. **Set up Python packages**:
   Ensure `__init__.py` files exist in:
   - `dags/__init__.py`
   - `dags/tasks/__init__.py`
   - `dags/utils/__init__.py`
   - `scripts/__init__.py`
   These can be empty files:
   ```bash
   mkdir -p dags/tasks dags/utils scripts
   touch dags/__init__.py dags/tasks/__init__.py dags/utils/__init__.py scripts/__init__.py
   ```

6. **Create directories**:
   ```bash
   mkdir -p huggingface_cache logs
   ```

7. **Build and start services**:
   ```bash
   docker-compose up -d
   ```
   This builds:
   - `custom-airflow:2.10.4` (from `Dockerfile`, if not already built)
   - `sentiment-analysis:latest` (from `Dockerfile.sentiment`)
   And starts all services (Airflow, MongoDB, sentiment analysis).

8. **Access Airflow**:
   - Web UI: `http://localhost:8080` (username: `airflow`, password: `airflow`)
   - Enable the `portfolio_news_pipeline` DAG.

## Configuration
- **Companies**: Edit `dags/utils/config.py` to add company names in the `COMPANIES` list.
- **MongoDB**: Uses `mongodb://mongo:27017/` (Docker service). Update `MONGO_URI` in `config.py` or `sentiment_analysis.py` if using a different host.
- **NewsAPI**: Set `NEWS_API_KEY` in `.env` or `config.py`.
- **Sentiment Model**: Uses `distilbert-base-uncased-finetuned-sst-2-english` with GPU acceleration. Modify `scripts/sentiment_analysis.py` to use a different Hugging Face model.
- **GPU**: Requires CUDA 12.1. Update `Dockerfile.sentiment` (e.g., to `nvidia/cuda:11.8.0-base-ubuntu20.04`) and `requirements.sentiment.txt` (e.g., `torch==2.4.1+cu118`) for other CUDA versions.

## Project Structure
```
portfolio_management/
├── dags/
│   ├── __init__.py                   # Python package marker
│   ├── news_pipeline.py              # Airflow DAG definition
│   ├── tasks/
│   │   ├── __init__.py              # Python package marker
│   │   ├── fetch_news.py            # Fetch weekly news
│   │   ├── store_news.py            # Store news in MongoDB
│   │   ├── check_historical_news.py  # Check historical news
│   │   ├── backfill_historical_news.py # Backfill historical news
│   └── utils/
│       ├── __init__.py              # Python package marker
│       ├── news_api_client.py       # NewsAPI client
│       ├── mongodb_client.py        # MongoDB client
│       └── config.py                # Configurations
├── scripts/
│   ├── __init__.py                  # Python package marker
│   └── sentiment_analysis.py        # Sentiment analysis script
├── docker-compose.yaml              # Docker services
├── Dockerfile.sentiment             # Sentiment analysis image
├── requirements.sentiment.txt       # Sentiment analysis dependencies
├── huggingface_cache/               # Hugging Face model cache
├── logs/                            # Airflow logs
└── README.md                        # Documentation
```

## Usage
- **Airflow Pipeline**:
  - Runs weekly (every Sunday at midnight), performing:
    - `fetch_news`: Fetches news articles via NewsAPI.
    - `store_news`: Stores articles in MongoDB.
    - `check_historical_data`: Backfills missing news for 52 weeks.
  - Stored in MongoDB (`portfolio_management.company_news`) with schema:
    - `company`: Company name (string)
    - `headline`: Article title (string)
    - `source`: News source name (string)
    - `url`: Article URL (string)
    - `published_at`: Publication date (string)
    - `week_start`: Start of the week (datetime)
    - `week_end`: End of the week (datetime)
    - `created_at`: Document creation timestamp (datetime)
    - `content`: Article content or description (string)
    - `sentiment`: Sentiment analysis result (object, e.g., `{"label": "POSITIVE", "score": 0.95}`)
- **Sentiment Analysis Service**:
  - Runs daily (every 24 hours) using GPU-accelerated PyTorch.
  - Processes articles with missing `sentiment` and non-empty `content`.
  - Updates MongoDB with `sentiment` field using `distilbert-base-uncased-finetuned-sst-2-english`.

## Troubleshooting
- **GPU issues**:
  - Verify drivers: `nvidia-smi`
  - Check NVIDIA Container Toolkit: `docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu20.04 nvidia-smi`
  - Check logs: `docker-compose logs sentiment-analysis`
  - Ensure `deploy` section in `docker-compose.yaml`:
    ```yaml
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    ```
- **MongoDB connection issues**:
  - Check logs: `docker-compose logs mongo`
  - Verify connectivity: `docker exec -it <mongo-container-id> mongosh mongodb://mongo:27017/ --eval "db.adminCommand('ping')"`
- **NewsAPI errors**: Verify `NEWS_API_KEY` in `.env`.
- **Sentiment analysis issues**:
  - Check logs: `docker-compose logs sentiment-analysis`
  - Verify GPU usage: `docker exec -it <sentiment-container-id> nvidia-smi`
  - Ensure internet for model downloads: `docker run -it sentiment-analysis:latest ping huggingface.co`
  - Check cache: `ls -l ./huggingface_cache`
- **Airflow webserver issues**:
  - Check logs: `docker-compose logs airflow-webserver`
  - Ensure PostgreSQL/Redis: `docker ps`
- **Build issues**:
  - Check build logs: `docker-compose logs sentiment-analysis`
  - Test manually: `docker build -t sentiment-analysis:latest -f Dockerfile.sentiment .`
- **DAG errors**:
  - Check logs: `docker-compose logs airflow-scheduler`
  - Verify `__init__.py` files
  - Test imports: `docker run -it -v $(pwd)/dags:/opt/airflow/dags custom-airflow:2.10.4 python /opt/airflow/dags/news_pipeline.py`