import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, f1_score
import mlflow
import warnings
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
warnings.filterwarnings('ignore')
import os
from io import StringIO
from sklearn import metrics

def train_test_split_example():
    iris = datasets.load_iris()

    X = iris.data[:, :2]
    y = iris.target
    
    train_X, test_X, train_Y, test_Y = train_test_split(X, y ,test_size=0.3 , random_state = 42)
    
    mod_dt = DecisionTreeClassifier(max_depth = 3, random_state = 1)
    mod_dt.fit(train_X,train_Y)
    prediction=mod_dt.predict(test_X)
    acc = metrics.accuracy_score(prediction,test_Y)
    print(f'The accuracy of the Decision Tree is {acc}')

    with mlflow.start_run():
        mlflow.log_metric('Accuracy Score', str(acc))


default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": datetime(2020, 10, 24),
    "schedule_interval": "@daily",
    "email": ["something@here.com"],
    "email_on_failure": False,
    "email_on_retry": False,
}

dag = DAG("python-test", default_args=default_args, schedule_interval="@once")

t1 = PythonOperator(
    task_id="pipeline", python_callable=train_test_split_example, dag=dag
)

t1
