#Dockerfile
FROM python:3.7 
# RUN pip3 install -r requirements.txt
RUN pip3 install 'apache-airflow' 'psycopg2' 'mlflow[extra]' 'scikit-learn' 'pandas' 'numpy'
EXPOSE 8000