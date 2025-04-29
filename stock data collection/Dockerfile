FROM python:3.12

WORKDIR /app


RUN apt-get update && apt-get install -y --no-install-recommends gcc  python3-dev && apt-get clean && rm -rf /var/lib/apt/lists/*


RUN python -m pip install --upgrade pip


COPY requirements.txt .


RUN pip install --no-cache-dir -r requirements.txt


COPY dags/ ./dags/


ENV AIRFLOW_HOME=/app/airflow 
ENV AIRFLOW__DATABASE__SQL_ALCHEMY_CONN=sqlite:////app/airflow/airflow.db 
ENV AIRFLOW__CORE__EXECUTOR=SequentialExecutor 
ENV AIRFLOW__CORE__LOAD_EXAMPLES=False 
ENV PYTHONUNBUFFERED=1

RUN airflow db migrate

# RUN airflow users create --username admin --firstname Admin --lastname User --role Admin --email admin@example.com --password admin

# CMD ["bash", "-c", "airflow scheduler & airflow webserver --port 8080"]

RUN echo '#!/bin/bash' > /app/start.sh && echo 'airflow users create --username admin --firstname Admin --lastname User --role Admin --email admin@example.com --password admin' >> /app/start.sh && echo 'airflow scheduler & airflow webserver --port 8080' >> /app/start.sh && chmod +x /app/start.sh
CMD ["/app/start.sh"]