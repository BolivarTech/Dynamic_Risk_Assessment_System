"""
ML monitoring 

Implementes the ML pipeline monitoring

By: Julian Bolivar
Version: 1.0.0
Date:  2023/06/20
Revision 1.0.0 (2023/06/20): Initial Release
"""

# Main System Imports
from argparse import ArgumentParser
import logging as log
import logging.handlers
import sys
import os
import platform
import mlflow

# Machine Learning imports
import pandas as pd

# Yaml file manager
import yaml

# Data Base Imports
import sqlite3 as db

# Get the running script's path
RUNNING_PATH = os.path.realpath(os.path.dirname(__file__)) 

# Main Logger
LOGHANDLER = None
LOGGER = None
LOGLEVEL_ = logging.INFO

# Load Configurion file
with open(os.path.join(RUNNING_PATH,'components','config.yaml')) as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

INPUT_FOLDER_PATH = os.path.join(RUNNING_PATH,'components',config['ingestion']['input_folder_path']) 
OUTPUT_FOLDER_PATH = os.path.join(RUNNING_PATH,'components',config['ingestion']['output_folder_path'])
PROD_DEPLOYMENT_PATH = os.path.join(RUNNING_PATH,'components',config['production']['prod_deployment_path'])
MODEL_PATH = os.path.join(RUNNING_PATH,'components',config['training']['output_model_path'])
DB_FILE = os.path.join(RUNNING_PATH,'components',config['database']['database_folder_path'],'pipeline_data.sqlite')


def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """

    parser = ArgumentParser(prog="fullprocess",
                            description="ML monitoring ")

    parser.add_argument("-v",
        "--version", 
        help="Show the script version",
        action="store_true",
        required=False
    )

    return parser.parse_args()

def main(args):
    """
    Run the main function

    args: command line arguments
    """

    global LOGGER

    move_to_next_step = False
    LOGGER.info("Launching automated monitoring")
    ##################Check and read new data
    #first, get ingested files
    #connect to a database, creating it if it doesn't exist 
    conn = db.connect(DB_FILE)
    LOGGER.info(f"Database Data File: {DB_FILE} (002)")
    if conn is not None:
        try: 
            ingestedfiles = pd.read_sql_query("select * from ingested_files",conn)
            LOGGER.info(f"Ingested Files table loaded from {DB_FILE} (003)")  
        except ValueError:
            # if exception occour Rollback
            conn.rollback()
            LOGGER.error(f"Can't read table 'ingested_files' in {DB_FILE} (005)")
        finally:
            # close out the connection
            conn.close()
            LOGGER.debug(f"Connection Closed (007)")
    else:
        LOGGER.error(f"Can't connect with {DB_FILE} (008)")

    #second, determine whether the source data folder has files that aren't listed in ingestedfiles.txt
    ingestedfiles['file'] = ingestedfiles['file'].apply(lambda x: os.path.basename(x))
    files = os.listdir(INPUT_FOLDER_PATH)
    files = [file for file in files if file not in ingestedfiles['file'].tolist()]
    
    ##################Deciding whether to proceed, part 1
    #if you found new data, you should proceed. otherwise, do end the process here
    if files !=[]:
        LOGGER.info("ingesting new files")
        # Ingest the files using the pipeline step
        _ = mlflow.run(
            os.path.join(RUNNING_PATH, "components"),
            "main",
            parameters={
                "steps": 'ingestion',
            }
        )
        move_to_next_step = True
    else:
        LOGGER.info("No new files - ending process")

    ##################Checking for model drift
    """There are two possible scenarios here:
    - We train a new model on new data and then compare the performance of the new model 
    and the existing model on the test data we set aside before.
    - We evaluate our existing model on the new data we have. If its performance falls 
    below its recorded performance on test data, we train a new model on the new data and deploy it.
    
    In the first case, we cannot explicitly say that the existing model drifted. 
    The model trained on new data simply performed better when evaluated on the same test data. 
    Whereas in the second scenario, the performance of our existing model degraded 
    on new data prompting us to train another model on the new data."""
    
    """You can manually change the dataset instead the latest score and 
    get an worse result only to make sure your code runs fine."""
    
    #check whether the score from the deployed model is different from the score from the model that uses the newest ingested data
    if move_to_next_step :
        # Score the new model using the pipeline step
        _ = mlflow.run(
            os.path.join(RUNNING_PATH, "components"),
            "main",
            parameters={
                "steps": 'scoring',
                "hydra_options": "training.output_model_path=" + config['production']['prod_deployment_path']
            }
        )
        #connect to a database, creating it if it doesn't exist 
        conn = db.connect(DB_FILE)
        LOGGER.info(f"Database Data File: {DB_FILE} (002)")
        if conn is not None:
            try: 
                modelscores = pd.read_sql_query("select * from model_score ORDER BY date ASC",conn)
                LOGGER.info(f"Score table loaded from {DB_FILE} (003)")  
            except ValueError:
                # if exception occour Rollback
                conn.rollback()
                LOGGER.error(f"Can't read table 'model_test_score' in {DB_FILE} (005)")
            finally:
                # close out the connection
                conn.close()
                LOGGER.debug(f"Connection Closed (007)")
        else:
            LOGGER.error(f"Can't connect with {DB_FILE} (008)")
    
        latest_score = modelscores.iloc[-2].values[1]
        new_score = modelscores.iloc[-1].values[1]
        LOGGER.info(f'latest score: {latest_score}, new score: {new_score} (001)')
        if new_score >= latest_score:
            move_to_next_step = False  # No model drift, keep existing model
            LOGGER.info('No model drift - ending process (001)')
    
    ##################Deciding whether to proceed, part 2
    #if you found model drift, you should proceed. otherwise, do end the process here
    if move_to_next_step:  # model drift, move to retraining
        LOGGER.info('training new model')
        # Retraings, Score, Deploy and Diagnoses the new model using the pipeline step
        _ = mlflow.run(
            os.path.join(RUNNING_PATH, "components"),
            "main",
            parameters={
                "steps": 'training,scoring,deployment,reporting',
            }
        )
    
    
if __name__ == '__main__':

    computer_name = platform.node()
    SCRIPT_NAME = "fullprocess"
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
    args = build_argparser()
    if args.version:
        print("Risk Pipeline v1.0.0")
    else:
        main(args)
    LOGGER.debug("Finished. (001)")
