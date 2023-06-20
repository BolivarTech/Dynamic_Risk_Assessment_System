"""
Model Training

MLFlow model training step

By: Julian Bolivar
Version: 1.0.0
Date:  2023-06-14
Revision 1.0.0 (2023-06-14): Initial Release
"""

# Main System Imports
from argparse import ArgumentParser
import logging as log
import logging.handlers
import sys
import os
import platform

# ML imports
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression

# Data Base Imports
import sqlite3 as db

# Get the running script's path
RUNNING_PATH = os.path.realpath(os.path.dirname(__file__)) 

# Main Logger
LOGHANDLER = None
LOGGER = None
LOGLEVEL_ = logging.INFO


def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """

    parser = ArgumentParser(prog="training",
                            description="Model Training")
    parser.add_argument("-i",
                        "--db_file", 
                        type=str,
                        help="Data ingested database",
                        default=os.path.join(RUNNING_PATH,'../../db/pipeline_data.sqlite'),
                        required=False)
    parser.add_argument("-o",
                        "--model_path", 
                        type=str,
                        help="Model save path",
                        default=os.path.join(RUNNING_PATH,'../../practicemodels'),
                        required=False)

    return parser.parse_args()


def segregate_dataset(dataset):
    """
    Eliminate features not used and segregate the dataset into X and y

    input (pandas dataframe): dataset to segregate
    output (pandas dataframe): X and y
    """

    # eliminate features not used for training
    features = ['lastmonth_activity','lastyear_activity','number_of_employees','exited']
    dataset = dataset[features]

    # data segregation
    predictors = features[:-1]
    target_variable = 'exited'
    X = dataset[predictors]
    y = dataset[target_variable]

    return X,y


def train_model(args):
    """
    Train a logistic regression model for churn classification
    input: script arguments
    output: trained model saved to disk
    """
    
    #use this logistic regression for training
    model = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                    intercept_scaling=1, l1_ratio=None, max_iter=100,
                    multi_class='auto', n_jobs=None, penalty='l2',
                    random_state=0, solver='liblinear', tol=0.0001, verbose=0,
                    warm_start=False)
    
    #connect to a database, creating it if it doesn't exist 
    conn = db.connect(args.db_file)
    LOGGER.info(f"Database Data File: {args.db_file} (002)")

    if conn is not None:
        try: 
            dataset = pd.read_sql_query("select * from ingested_data",conn)
            LOGGER.info(f"Ingested Data table loaded from {args.db_file} (003)")  
        except ValueError:
            # if exception occour Rollback
            conn.rollback()
            LOGGER.error(f"Can't read table 'ingested_data' in {args.db_file} (005)")
        finally:
            # close out the connection
            conn.close()
            LOGGER.debug(f"Connection Closed (007)")
    else:
        LOGGER.error(f"Can't connect with {args.db_file} (008)")

    X,y = segregate_dataset(dataset)
    
    # fit the logistic regression to your data
    model.fit(X,y)
    
    # write the trained model to your workspace in a file called trainedmodel.pkl
    savingpath = os.path.join(args.model_path,'trainedmodel.pkl')
    with open(savingpath, 'wb') as file:
        pickle.dump(model, file)


def main(args):
    """
    Run the main function

    args: command line arguments
    """

    global LOGGER

    train_model(args)


if __name__ == '__main__':

    computer_name = platform.node()
    SCRIPT_NAME = "training"
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
    main(args)
    LOGGER.debug("Finished. (001)")
