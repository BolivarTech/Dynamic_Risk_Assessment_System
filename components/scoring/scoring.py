"""
Model Scoring

MLFLow model scoring step

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
from datetime import datetime as dt

# ML imports
import pickle
from sklearn import metrics
import pandas as pd

# Data Base Imports
import sqlite3 as db

# Get the running script path
RUNNING_PATH = os.path.realpath(os.path.dirname(__file__)) 

# adding training directory to the system path
sys.path.insert(0, os.path.join(RUNNING_PATH, '../training'))

# Imports from other libraries
from training import segregate_dataset


# Main Logger
LOGHANDLER = None
LOGGER = None
LOGLEVEL_ = logging.INFO


def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """

    parser = ArgumentParser(prog="scoring",
                            description="Model Scoring")

    parser.add_argument("-d",
                        "--db_file", 
                        type=str,
                        help="Data ingested database",
                        default=os.path.join(RUNNING_PATH,'../../db/pipeline_data.sqlite'),
                        required=False)
    
    parser.add_argument("-m",
        "--model_file", 
        type=str,
        help="Model saved file",
        default=os.path.join(RUNNING_PATH,'../../practicemodels/trainedmodel.pkl'),
        required=False
    )
    
    parser.add_argument("-t",
        "--data_test_file", 
        type=str,
        help="Data test File",
        default=os.path.join(RUNNING_PATH,'../../testdata/testdata.csv'),
        required=False
    )


    return parser.parse_args()


def score_model(args):
    """
    Perform the F1 model scoring using the test data and save it on the db table
    'model_scores' and the last one is stored at the model's path in the
    'latestscore.txt' file.

    inputs:
        args [dict]: parsed system arguments

    outputs:
       none 
    """

    # import test dataset from csv file
    testdata = pd.read_csv(args.data_test_file)

    # load trained model    
    with open(args.model_file, 'rb') as file:
        model = pickle.load(file)
        LOGGER.info(f"Model {args.model_file} loaded (001)")

    # segregate test dataset
    X, y = segregate_dataset(testdata)

    # evaluate model on test set
    yhat = model.predict(X)
    score = metrics.f1_score(y, yhat)

    # upate score table
    #connect to a database, creating it if it doesn't exist 
    conn = db.connect(args.db_file)
    LOGGER.info(f"Database Data File: {args.db_file} (001)")
    if conn is not None:
        try: 
            # get current time
            now = dt.now().strftime("%Y-%m-%d %H:%M:%S")
            # Create Score Data Frame
            score_reg = {'date': [now,], 'score': [score,]}
            scores_df = pd.DataFrame(score_reg)
            # Save score record into database
            scores_df.to_sql("model_score", conn, if_exists='append', index=False)
            LOGGER.info(f"Score recorded in 'model_score' table into {args.db_file} (001)")
        except ValueError as err:
            # if exception occour Rollback
            conn.rollback()
            LOGGER.error(f"Can't update table 'model_score' in {args.db_file} (001)\n{err}")
        else:
            # commit the transaction
            conn.commit()
            LOGGER.debug(f"Transactions commited (001)")
        finally:
            # close out the connection
            conn.close()
            LOGGER.debug(f"Connection Closed (001)")
    else:
        LOGGER.error(f"Can't connect with {args.output_file} (001)")

    # save as latest score on file
    scorespath = os.path.join(os.path.realpath(os.path.dirname(args.model_file)), 'latestscore.txt')
    with open(scorespath, 'w') as file:
        file.write(str(score))


def main(args):
    """
    Run the main function

    args: command line arguments
    """

    global LOGGER

    score_model(args)


if __name__ == '__main__':

    computer_name = platform.node()
    SCRIPT_NAME = "scoring"
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
