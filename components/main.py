"""
Main Pipeline

Main Pipeline Script

By: Julian Bolivar
Version: 1.0.0
Date:  2023/06/19
Revision 1.0.0 (2023/06/19): Initial Release
"""

# Main System Imports
import logging as log
import logging.handlers
import sys
import os
import platform
import mlflow
import hydra
from omegaconf import DictConfig
import tempfile


# Get the running script's path
RUNNING_PATH = os.path.realpath(os.path.dirname(__file__)) 

# Main Logger
LOGHANDLER = None
LOGGER = None
LOGLEVEL_ = logging.INFO

# The steps will be executed on this order
_steps = [
    "ingestion",
    "training",
    "scoring",
    "deployment",
    "reporting"
]

# This automatically reads in the configuration
@hydra.main(config_path=".", config_name='config', version_base=None)
def main(config: DictConfig):
    """
    Run the main function

    args: command line arguments
    """

    global LOGGER

    # Get the path at the root of the MLflow project
    hydra_root_path = hydra.utils.get_original_cwd()

    # Steps to execute
    steps_par = config['main']['steps']
    active_steps = steps_par.split(",") if steps_par != "all" else _steps

    # Move to a temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        if "ingestion" in active_steps:
            # Ingest the files
            _ = mlflow.run(
                os.path.join(hydra_root_path, "ingestion"),
                "main",
                parameters={
                    "input_path": os.path.join(hydra_root_path, config["ingestion"]["input_folder_path"]),
                    "out_file": os.path.join(hydra_root_path, config["ingestion"]["output_folder_path"], "finaldata.csv"),
                    "record_file": os.path.join(hydra_root_path, config["ingestion"]["output_folder_path"], "ingestedfiles.txt"),
                    "db_file": os.path.join(hydra_root_path, config["database"]["database_folder_path"], "pipeline_data.sqlite")
                }
            )
        if "training" in active_steps:
            # Train the Model
            _ = mlflow.run(
                os.path.join(hydra_root_path, "training"),
                "main",
                parameters={
                    "model_path": os.path.join(hydra_root_path, config["training"]["output_model_path"]),
                    "db_file": os.path.join(hydra_root_path, config["database"]["database_folder_path"], "pipeline_data.sqlite")
                }
            )
        if "scoring" in active_steps:
            # Score the model
            _ = mlflow.run(
                os.path.join(hydra_root_path, "scoring"),
                "main",
                parameters={
                    "model_file": os.path.join(hydra_root_path, config["training"]["output_model_path"], "trainedmodel.pkl"),
                    "data_test_file": os.path.join(hydra_root_path, config["diagnostics"]["test_data_path"], "testdata.csv"),
                    "db_file": os.path.join(hydra_root_path, config["database"]["database_folder_path"], "pipeline_data.sqlite")
                }
            )
        if "deployment" in active_steps:
            # Deploy the model
            _ = mlflow.run(
                os.path.join(hydra_root_path, "deployment"),
                "main",
                parameters={
                    "model_path": os.path.join(hydra_root_path, config["training"]["output_model_path"]),
                    "record_file": os.path.join(hydra_root_path, config["ingestion"]["output_folder_path"], "ingestedfiles.txt"),
                    "deploy_path": os.path.join(hydra_root_path, config["production"]["prod_deployment_path"])
                }
            )
        if "reporting" in active_steps:
            # Generate Model Report
            _ = mlflow.run(
                os.path.join(hydra_root_path, "reporting"),
                "main",
                parameters={
                    "model_file": os.path.join(hydra_root_path, config["production"]["prod_deployment_path"], "trainedmodel.pkl"),
                    "test_data_file": os.path.join(hydra_root_path, config["diagnostics"]["test_data_path"], "testdata.csv"),
                    "db_file": os.path.join(hydra_root_path, config["database"]["database_folder_path"], "pipeline_data.sqlite")
                }
            )


if __name__ == '__main__':

    computer_name = platform.node()
    SCRIPT_NAME = "main"
    loggPath = os.path.join(".","log")
    if not os.path.isdir(loggPath):
        try:
            # mode forced due security
            MODE = 0o770
            os.mkdir(loggPath, mode=MODE)
        except OSError as error:
            print(error)
            sys.exit(-1)
    LogFileName = os.path.join(loggPath,
                               computer_name + '-' + SCRIPT_NAME + '.log')
    # Configure the logger
    LOGGER = log.getLogger(SCRIPT_NAME)  # Get Logger
    # Add the log message file handler to the logger
    LOGHANDLER = log.handlers.RotatingFileHandler(LogFileName,
                                                  maxBytes=10485760,
                                                  backupCount=10)
    # Logger Formater
    logFormatter = log.Formatter(fmt='%(asctime)s - %(name)s - %(levelname)s: %(message)s',
                                datefmt='%Y/%m/%d %H:%M:%S')
    LOGHANDLER.setFormatter(logFormatter)
    # Add handler to logger
    if 'LOGHANDLER' in globals():
        LOGGER.addHandler(LOGHANDLER)
    else:
        LOGGER.debug("logHandler NOT defined (001)")
    # Set Logger Lever
    LOGGER.setLevel(LOGLEVEL_)
    # Start Running
    LOGGER.debug("Running... (001)")
    main()
    LOGGER.debug("Finished. (001)")
