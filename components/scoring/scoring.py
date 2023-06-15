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

# ML imports
import pickle
from sklearn import metrics
import pandas as pd

# Data Base Imports
import sqlite3 as db

# adding training directory to the system path
sys.path.insert(0, '../training')

# Get the running script path
# os.path.realpath(os.path.dirname(__file__))

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
                        default='../../db/pipeline_data.sqlite',
                        required=False)
    
    parser.add_argument("-m",
        "--model_path", 
        type=str,
        help="Model saved path",
        default='../../practicemodels',
        required=False
    )
    
    parser.add_argument("-t",
        "--data_test_path", 
        type=str,
        help="Data test path",
        default='../../testdata',
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
    testfile = os.path.join(args.data_test_path, 'testdata.csv')
    testdata = pd.read_csv(testfile)

    # load trained model
    modelpath = os.path.join(args.model_path, 'trainedmodel.pkl')
    with open(modelpath, 'rb') as file:
        model = pickle.load(file)

    # segregate test dataset
    X, y = segregate_dataset(testdata)

    # evaluate model on test set
    yhat = model.predict(X)
    score = metrics.f1_score(y, yhat)

    # save as latest score
    scorespath = os.path.join(args.model_path, 'latestscore.txt')
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
