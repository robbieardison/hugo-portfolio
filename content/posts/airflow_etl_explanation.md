---
title: "Building a Simple ETL Pipeline with Apache Airflow: A Step-by-Step Guide"
date: 2023-07-01
tags: ["airflow", "etl"]
draft: false
categories:
  - Data Engineering
thumbnail: ./images/apache_airflow_logo.png
toc: true
---

## Title: Building a Simple ETL Pipeline with Apache Airflow: A Step-by-Step Guide

### Introduction

Apache Airflow is an open-source platform for orchestrating complex data workflows. In this blog post, I'll guide you through the process of building a simple ETL pipeline using Apache Airflow. Our example will focus on extracting data from a source, transforming it, and loading it into a destination.

### Prerequisites

Before you start, make sure you have Apache Airflow installed. You can install it using:

```bash
pip install apache-airflow
```

### Step 1: Setting up your DAG (Directed Acyclic Graph)

In Airflow, workflows are defined as DAGs. Create a new Python file for your DAG, e.g., `etl_pipeline.py`. Define your DAG as follows:

```python
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python_operator import PythonOperator

# Define default_args and DAG
default_args = {
    'owner': 'your_name',
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'etl_pipeline',
    default_args=default_args,
    schedule_interval=timedelta(days=1),
)
```

### Step 2: Define your ETL functions

Now, define the functions for extracting, transforming, and loading data. For simplicity, let's use Python functions:

```python
def extract_data(**kwargs):
    # Your code to extract data
    pass

def transform_data(**kwargs):
    # Your code to transform data
    pass

def load_data(**kwargs):
    # Your code to load data
    pass
```

### Step 3: Create task instances

Create task instances for each ETL step using `PythonOperator`:

```python
extract_task = PythonOperator(
    task_id='extract_task',
    python_callable=extract_data,
    provide_context=True,
    dag=dag,
)

transform_task = PythonOperator(
    task_id='transform_task',
    python_callable=transform_data,
    provide_context=True,
    dag=dag,
)

load_task = PythonOperator(
    task_id='load_task',
    python_callable=load_data,
    provide_context=True,
    dag=dag,
)
```

### Step 4: Define the task dependencies

Set up the dependencies between tasks:

```python
extract_task >> transform_task >> load_task
```

### Step 5: Run your ETL pipeline

Save your file and start the Airflow scheduler and web server:

```bash
airflow scheduler
airflow webserver
```

Visit `http://localhost:8080` in your browser and trigger your DAG to see the ETL pipeline in action.

### Conclusion

Congratulations! You've just created a simple ETL pipeline using Apache Airflow. This example provides a foundation for building more complex workflows to meet your specific data processing needs.

Feel free to customize the example based on your data sources, transformations, and destinations. Explore Airflow's rich features to enhance and scale your ETL processes. Happy data orchestrating!