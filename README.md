# xai-analytics-pipeline
Analytical pipeline that generates explainability reports using SHAP and LIME as an analysis base, allows the integration of an artificial intelligence model and activates the pipeline automatically so that it generates the corresponding graphs and metrics.

# WSL
`wsl -u serveradmin -d Ubuntu-22.04`


# switch to python environment
`source venv/bin/activate`

# enviroment set up
`export AIRFLOW_HOME=/mnt/d/Users/Javier/Documents/Proyectos/2023/analytic-pipeline/airflow`

# starting airlofw WEB UI
`airflow webserver`

# starting airlofw scheduler
`airflow scheduler`