# Dynamic Risk Assessment System

![Conda](https://img.shields.io/conda/pn/conda-forge/python)
![licence](https://img.shields.io/badge/language-Python-brightgreen.svg?style=flat-square)

On this project a Dynamic Machine Learning Risk Assessment pipeline is
implemented following the ML DevOps standards.

In this project you will build such a pipeline.

- GitHub release 1.0.1: https://github.com/BolivarTech/Dynamic_Risk_Assessment_System.git

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

An REST API interface was implemented to allow monitoring and 
