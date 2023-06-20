# Dynamic Risk Assessment System

![Conda](https://img.shields.io/conda/pn/conda-forge/python)
![licence](https://img.shields.io/badge/language-Python-brightgreen.svg?style=flat-square)

On this project a Dynamic Machine Learning Risk Assessment pipeline is
implemented following the ML DevOps standards.

In this project you will build such a pipeline.

- GitHub release 1.0.0: https://github.com/BolivarTech/Dynamic_Risk_Assessment_System.git

The following tools are used:

- [MLflow](https://www.mlflow.org) for reproduction and management of pipeline processes.
- [Hydra](https://hydra.cc) for configuration management.
- [Conda](https://docs.conda.io/en/latest) for environment management.
- [Pandas](https://pandas.pydata.org) for data analysis.
- [Scikit-Learn](https://scikit-learn.org/stable) for data modeling.
- [SQlite](https://www.sqlite.org) for data tracking and storing
- [Flask](https://flask.palletsprojects.com/) for REST API interface

The final goal of the pipeline is to produce the optimal inference artifact 
to predict risk assessment, monitoring model's F1 drifts and when the drift is 
detected retraining and deploying the new model.

Aditional it implements a REST API interface was implemented to allow monitoring and download performance
report on PDF file

## How to Use This Project

1. Install the [dependencies](#dependencies).
2. Run the pipeline as explained in [the dedicated section](#how-to-run-the-pipeline).

### Dependencies

In order to set up the main environment from which everything is launched you need to install [conda](https://docs.conda.io/en/latest/) and the following sets everything up:

```bash
# Clone repository
git clone https://github.com/BolivarTech/Dynamic_Risk_Assessment_System.git
cd Dynamic_Risk_Assessment_System

# Create new environment
conda env create -f environment.yml

# Activate environment
conda activate risk_pipeline
```

All step/component dependencies are handled by MLflow using the dedicated `conda.yaml` environment definition files.

### How to Run the Pipeline

There are multiple ways of running this pipeline, for instance:

- Local execution or execution from cloned source code of the complete pipeline.
- Local execution of selected pipeline steps.
- Full automatinc Pipeline monitoring and retraing process
- API monitoring access.

In the following, some example commands that show how these approaches work are listed:

```bash
# Go to the root project level, where main.py is
cd Dynamic_Risk_Assessment_System

# Local execution of the entire pipeline
mlflow run ./components

# To excetute specifis pipelines steps
# Step names are ingestion, training, scoring, deployment, reporting
mlflow run ./components -P steps="ingestion,training,scoring"

# To exceute the pipeline and monitor it performance, retraining and deployiment
python3 fullprocess.py  
```

To run this pipeline a cron job should be installed, and example is provided on 
`cronjob.txt` to run it every 10 minutes, adjust it to your required needs.

```bash
*/10 * * * * /home/jbolivarg/udacity/Dynamic_Risk_Assessment_System/cron_run.sh
```

To activate the conda env on the cron job a bash script `cron_run.sh` is provided,
adjust the paths to the running environment.


#### API Interface

```bash
# To excecute the API server
mlflow run ./components/app

# Call all the APIs from the command line a MLFlow entry point called 'apicalls'
# was defined to produce all the reports
mlflow run -e apicalls ./components/app

```

On your internet browser you can download the pipeline performance report usinng
the follow URL: <br><br> 
`http://[SERVER IP ADDRESS]:8000/download`

## Authorship

[Julian Bolivar](https://www.linkedin.com/in/jbolivarg), 2023.  
