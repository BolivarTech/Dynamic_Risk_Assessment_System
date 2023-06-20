"""
Pipeline report generator

Generate the ML pipeline performance report

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

# Machine Learning imports
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import ast

# Data Base Imports
import sqlite3 as db

# Get the running script's path
RUNNING_PATH = os.path.realpath(os.path.dirname(__file__)) 

# adding training directory to the system path
sys.path.insert(0, os.path.join(RUNNING_PATH, '../diagnostics'))

from diagnostics import (model_predictions, dataframe_summary, missing_data, 
                        execution_time, outdated_packages_list)


# Main Logger
LOGHANDLER = None
LOGGER = None
LOGLEVEL_ = logging.INFO


def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """

    parser = ArgumentParser(prog="reporting",
                            description="Pipeline report generator")


    parser.add_argument("-t",
        "--test_data_file", 
        type=str,
        help="Data test file",
        default=os.path.join(RUNNING_PATH,'../../testdata/testdata.csv'),
        required=False
    )

    parser.add_argument("-m",
        "--model_file", 
        type=str,
        help="Model File",
        default=os.path.join(RUNNING_PATH,'../../production_deployment/trainedmodel.pkl'),
        required=False
    )

    parser.add_argument("-d",
                        "--db_file", 
                        type=str,
                        help="Database",
                        default=os.path.join(RUNNING_PATH,'../../db/pipeline_data.sqlite'), 
                        required=False)
    
    return parser.parse_args()


def save_multi_image(filename, plt):
    """
    Save images to PDF

    :param filename: (str) File where to save report
    :param plt: (plot) Matplot plots
    """
    pp = PdfPages(filename)
    fig_nums = plt.get_fignums()
    figs = [plt.figure(n) for n in fig_nums]
    for fig in figs:
        fig.savefig(pp, format='pdf')
    pp.close()


def score_model(args):
    """
    calculate a confusion matrix using the test data and the deployed model

    :param args: (dict) command line parameters

    """ 
        
    # collect test dataset
    dataset = pd.read_csv(args.test_data_file)
    # perform prediction
    yhat = model_predictions(args.model_file, args.test_data_file, args.db_file, LOGGER)
    # calculate confusion matrix
    y = dataset['exited']
    cm = metrics.confusion_matrix(y, yhat)

    # Create cm plot
    f, ax = plt.subplots(figsize=(5,4))
    sns.heatmap(cm, annot=True,cmap='viridis', fmt='d', linewidths=.5, annot_kws={"fontsize":15})
    plt.xlabel('Predicted Class', fontsize = 15)
    ax.xaxis.set_ticklabels(['Not Churned', 'Churned'])
    plt.ylabel('True Class', fontsize = 15)
    ax.yaxis.set_ticklabels(['Not Churned', 'Churned'])
    plt.title('Confusion matrix', fontsize = 20)
    
    # write the confusion matrix to the workspace
    model_output_path = os.path.realpath(os.path.dirname(args.model_file))
    savepath = os.path.join(model_output_path,'confusionmatrix.png')
    f.savefig(savepath)

    # Additional Statistics
    # compute classification report
    cr = metrics.classification_report(y, yhat, output_dict=True)
    # Collect statistics
    statistics = dataframe_summary(args.db_file, LOGGER)
    missingdata = missing_data(args.db_file, LOGGER)
    timings = execution_time()
    dependencies = outdated_packages_list()
    # collect ingested files
    #connect to a database, creating it if it doesn't exist 
    conn = db.connect(args.db_file)
    LOGGER.info(f"Database Data File: {args.db_file} (002)")

    if conn is not None:
        try: 
            ingestedfiles = pd.read_sql_query("select * from ingested_files",conn)
            LOGGER.info(f"Ingested Files table loaded from {args.db_file} (003)")  
        except ValueError:
            # if exception occour Rollback
            conn.rollback()
            LOGGER.error(f"Can't read table 'ingested_files' in {args.db_file} (005)")
        finally:
            # close out the connection
            conn.close()
            LOGGER.debug(f"Connection Closed (007)")
    else:
        LOGGER.error(f"Can't connect with {args.db_file} (008)")

    # Produce pdf report
    # 1- list of ingested files
    ingestedfiles['file'] = ingestedfiles['file'].apply(lambda x: os.path.basename(x))
    ingestedfiles.rename(columns = {'file':'Ingested File'}, inplace = True)
    col_names = ingestedfiles.columns.tolist()
    data = ingestedfiles.values
    rowLabels = ingestedfiles.index.tolist()
    # Plot table
    fig, ax = plt.subplots(1, figsize=(5,5))
    plt.title('Ingested files', fontsize = 30)
    ax.axis('off')
    table = plt.table(cellText=data, colLabels=col_names, loc='center',colLoc='right', rowLabels=rowLabels)
    plt.tight_layout()

    # 2- summary statistics
    col_names = ['lastmonth_activity','lastyear_activity','number_of_employees','exited']
    data = np.array(statistics).reshape(3,4)
    # Plot table
    fig, ax = plt.subplots(1, figsize=(10,2))
    plt.title('Summary statistics', fontsize = 20)
    ax.axis('off')
    table = plt.table(cellText=data, colLabels=col_names, loc='center',colLoc='right',rowLabels=['mean','median','std'])

    # 3- Confusion matrix
    # Already created above

    # 4- classification report
    df = pd.DataFrame(cr).transpose()
    col_names = df.columns.tolist()
    data = df.values
    rowLabels = df.index.tolist()
    # Plot table
    fig, ax = plt.subplots(1, figsize=(10,5))
    ax.axis('off')
    table = plt.table(cellText=data, colLabels=col_names, loc='center',colLoc='right',rowLabels=rowLabels)
    plt.title('Classification report', fontsize = 20)
    plt.tight_layout()

    # 5- Missing data
    df = pd.DataFrame(data=missingdata, index = dataset.columns.tolist(), columns=['missing data'])
    col_names = df.columns.tolist()
    data = df.values
    rowLabels = df.index.tolist()
    # Plot table
    fig, ax = plt.subplots(1, figsize=(4,6))
    ax.axis('off')
    table = plt.table(cellText=data, colLabels=col_names, loc='center',colLoc='right',rowLabels=rowLabels)
    plt.title('Missing data', fontsize = 20)
    plt.tight_layout()

    # 6- Timing of execution
    timing = pd.DataFrame(timings, columns=['Duration (sec)'])
    col_names = timing.columns.tolist()
    data = timing.values
    rowLabels = ["Ingestion step", 'Training step']
    # Plot table
    fig, ax = plt.subplots(1, figsize=(4,4))
    plt.title('Execution time', fontsize = 20)
    ax.axis('off')
    table = plt.table(cellText=data, colLabels=col_names, loc='center',colLoc='right', rowLabels=rowLabels)
    plt.tight_layout()

    # 7- dependencies status
    col_names = dependencies.columns.tolist()
    data = dependencies.values
    rowLabels = dependencies.index.tolist()
    # Plot table
    fig, ax = plt.subplots(1, figsize=(5,5))
    plt.title('dependencies status', fontsize = 20)
    ax.axis('off')
    table = plt.table(cellText=data, colLabels=col_names, loc='center',colLoc='right',rowLabels=rowLabels)
    plt.tight_layout()

    filename = os.path.join(model_output_path,"report.pdf")  
    save_multi_image(filename,plt)


def main(args):
    """
    Run the main function

    args: command line arguments
    """

    global LOGGER

    score_model(args)


if __name__ == '__main__':

    computer_name = platform.node()
    SCRIPT_NAME = "reporting"
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
