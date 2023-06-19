"""
Model and data diagnostics

Perform the model and the data diagnostics and generate reports

By: Julian Bolivar
Version: 1.0.0
Date:  2023/06/18
Revision 1.0.0 (2023/06/18): Initial Release
"""

# Main System Imports
from argparse import ArgumentParser
import logging as log
import logging.handlers
import sys
import os
import platform
import pickle
from io import StringIO
import subprocess
import timeit
from datetime import datetime as dt

# Data Base Imports
import sqlite3 as db

# Machine learning imports
import pandas as pd
import numpy as np
from sklearn import metrics

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

    parser = ArgumentParser(prog="diagnostics",
                            description="Model and data diagnostics")


    parser.add_argument("-m",
        "--model_path", 
        type=str,
        help="Path to deploymed model file",
        default=os.path.join(RUNNING_PATH,'../../production_deployment/trainedmodel.pkl'),
        required=False
    )

    parser.add_argument("-t",
        "--test_data_file", 
        type=str,
        help="Path to test data file",
        default=os.path.join(RUNNING_PATH,'../../testdata/testdata.csv'),
        required=False
    )

    parser.add_argument("-o",
        "--output_path", 
        type=str,
        help="Path where the report is saved",
        default=os.path.join(RUNNING_PATH,'../../ingesteddata'),
        required=False
    )

    parser.add_argument("-d",
        "--db_path", 
        type=str,
        help="Path to database",
        default=os.path.join(RUNNING_PATH,'../../db/pipeline_data.sqlite'),
        required=False
    )

    return parser.parse_args()


def model_predictions(model_path, test_data_path, db_path):
    """
    read the deployed model and a test dataset, calculate predictions F1 Score
    and estor it on the database
    
    :param model_path: (str) current state
    :param test_data_path: (str) Add noise using the epsilon-greedy policy
    :param db_path: (str) Add noise using the epsilon-greedy policy
    :return: list of predictions from deployed model
    """
    # load test dataset
    dataset = pd.read_csv(test_data_path)

    # collect deployed model
    with open(model_path, 'rb') as file:
        model = pickle.load(file)

    # segregate test dataset
    X, y = segregate_dataset(dataset)

    # evaluate model on test set
    yhat = model.predict(X)

    # Verify data input and output length
    if len(yhat) != len(y):
        LOGGER.error(f"length for input ({len(y)}) and output ({len(yhat)}) must be the same (001)")
    assert len(yhat) == len(y), "length for input and output must be the same"
    
    # Score model on test set
    score = metrics.f1_score(y, yhat)

    # upate score table
    #connect to a database, creating it if it doesn't exist 
    conn = db.connect(db_path)
    LOGGER.info(f"Database Data File: {db_path} (001)")
    if conn is not None:
        try: 
            # get current time
            now = dt.now().strftime("%Y-%m-%d %H:%M:%S")
            # Create Score Data Frame
            score_reg = {'date': [now,], 'score': [score,]}
            scores_df = pd.DataFrame(score_reg)
            # Save score record into database
            scores_df.to_sql("model_test_score", conn, if_exists='append', index=False)
            LOGGER.info(f"Score recorded in 'model_score' table into {db_path} (001)")
        except ValueError as err:
            # if exception occour Rollback
            conn.rollback()
            LOGGER.error(f"Can't update table 'model_test_score' in {db_path} (001)\n{err}")
        else:
            # commit the transaction
            conn.commit()
            LOGGER.debug(f"Transactions commited (001)")
        finally:
            # close out the connection
            conn.close()
            LOGGER.debug(f"Connection Closed (001)")
    else:
        LOGGER.error(f"Can't connect with {db_path} (001)")

    return yhat


def dataframe_summary(db_path):
    """
    Calculate summary statistics on the dataset columns

    :param db_path: (str) Add noise using the epsilon-greedy policy

    :return: list with dataframe's means, medians and stddevs 
    """

    #connect to a database, creating it if it doesn't exist 
    conn = db.connect(db_path)
    LOGGER.info(f"Database Data File: {db_path} (002)")

    if conn is not None:
        try: 
            dataset = pd.read_sql_query("select * from ingested_data",conn)
            LOGGER.info(f"Ingested Data table loaded from {db_path} (003)")  
        except ValueError:
            # if exception occour Rollback
            conn.rollback()
            LOGGER.error(f"Can't read table 'ingested_data' in {db_path} (005)")
        finally:
            # close out the connection
            conn.close()
            LOGGER.debug(f"Connection Closed (007)")
    else:
        LOGGER.error(f"Can't connect with {db_path} (008)")

    # Select numeric columns
    numeric_col_index = np.where(dataset.dtypes != object)[0]
    numeric_col = dataset.columns[numeric_col_index].tolist()

    # compute statistics per numeric column
    means = dataset[numeric_col].mean(axis=0).tolist()
    medians = dataset[numeric_col].median(axis=0).tolist()
    stddevs = dataset[numeric_col].std(axis=0).tolist()

    statistics = means
    statistics.extend(medians)
    statistics.extend(stddevs)

    return statistics


def missing_data(db_path):
    """
    calculate missing data on the dataset
    return % of missing data per column

    :param db_path: (str) Add noise using the epsilon-greedy policy

    :return: list with dataframe's % missing data per column 
    """

    #connect to a database, creating it if it doesn't exist 
    conn = db.connect(db_path)
    LOGGER.info(f"Database Data File: {db_path} (002)")

    if conn is not None:
        try: 
            dataset = pd.read_sql_query("select * from ingested_data",conn)
            LOGGER.info(f"Ingested Data table loaded from {db_path} (003)")  
        except ValueError:
            # if exception occour Rollback
            conn.rollback()
            LOGGER.error(f"Can't read table 'ingested_data' in {db_path} (005)")
        finally:
            # close out the connection
            conn.close()
            LOGGER.debug(f"Connection Closed (007)")
    else:
        LOGGER.error(f"Can't connect with {db_path} (008)")

    # compute missing data % per column
    missing_data = dataset.isna().sum(axis=0)
    missing_data /= len(dataset) *100

    return missing_data.tolist()


def execution_time():
    """
    calculate timing of training.py and ingestion.py

    :return: list with each one running time on seconds 
    """
    
    timing_measures = []

    # timing ingestion step
    start_time = timeit.default_timer()
    os.system(f"python {os.path.join(RUNNING_PATH,'../ingestion/ingestion.py')}")
    end_time = timeit.default_timer()
    duration_step = end_time - start_time
    timing_measures.append(duration_step)

    # timing ingestion step
    start_time = timeit.default_timer()
    os.system(f"python {os.path.join(RUNNING_PATH,'../training/training.py')}")
    end_time = timeit.default_timer()
    duration_step = end_time - start_time
    timing_measures.append(duration_step)

    return timing_measures


def execute_cmd(cmd):
    """
    execute a pip list type cmd
    
    :param cmd: (list) pip list type of cmd
    
    :return: output of cmd in dataframe format
    """

    a = subprocess.Popen(cmd, stdout=subprocess.PIPE) 
    b = StringIO(a.communicate()[0].decode('utf-8'))
    df = pd.read_csv(b, sep="\s+")
    df.drop(index=[0], axis=0, inplace=True)
    df = df.set_index('Package')
    return df

##################Function to check dependencies
def outdated_packages_list():
    """get a list of dependencies and versions
    :inputs: None
    :return: dataframe with list of outdated dependencies, 
             version as per requirements.txt file,
             and latest version available
    """

    # collect outdated dependencies (for current virtual env)
    cmd = ['pip', 'list', '--outdated']
    df = execute_cmd(cmd)
    df.drop(['Version','Type'], axis=1, inplace=True)

    # collect all dependencies (for current virtual env)
    cmd = ['pip', 'list']
    df1 = execute_cmd(cmd)
    df1 = df1.rename(columns = {'Version':'Latest'})

    # collect dependencies as per requirements.txt file
    requirements = pd.read_csv('requirements.txt', sep='==', header=None, names=['Package','Version'], engine='python')
    requirements = requirements.set_index('Package')

    # assemble target and latest versions for requirements.txt dependencies
    dependencies = requirements.join(df1)
    for p in df.index:
        if p in dependencies.index:
            dependencies.at[p, 'Latest'] = df.at[p,'Latest']
    
    # keep only outdated dependencies (ie latest version exists)
    dependencies.dropna(inplace=True)

    return dependencies


def main(args):
    """
    Run the main function

    args: command line arguments
    """

    global LOGGER

    _ = model_predictions(args.model_path, args.test_data_file, args.db_path)
    _ = dataframe_summary(args.db_path)
    _ = missing_data(args.db_path)
    _ = execution_time()
    _ = outdated_packages_list()


if __name__ == '__main__':

    computer_name = platform.node()
    SCRIPT_NAME = "diagnostics"
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
