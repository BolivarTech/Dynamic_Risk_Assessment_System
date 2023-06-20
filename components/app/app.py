"""
API Interface

Implements the REST API interface

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

# ML imports
import pandas as pd

# API imports
from flask import Flask, session, jsonify, request, send_from_directory

# Yaml file manager
import yaml

# Get the running script's path
RUNNING_PATH = os.path.realpath(os.path.dirname(__file__)) 

# adding training directory to the system path
sys.path.insert(0, os.path.join(RUNNING_PATH, '../diagnostics'))
sys.path.insert(0, os.path.join(RUNNING_PATH, '../scoring'))

from diagnostics import (model_predictions, dataframe_summary, missing_data, 
                        execution_time, outdated_packages_list)

from scoring import score_model

# Main Logger
LOGHANDLER = None
LOGGER = None
LOGLEVEL_ = logging.INFO

# Falsk Environment
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

# Load Configurion file
with open(os.path.join(RUNNING_PATH,'..','config.yaml')) as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

dataset_csv_path = os.path.join(RUNNING_PATH,'..',config['ingestion']['output_folder_path']) 
db_file = os.path.join(RUNNING_PATH,'..',config['database']['database_folder_path'],'pipeline_data.sqlite')
model_file = os.path.join(RUNNING_PATH,'..',config['production']['prod_deployment_path'],'trainedmodel.pkl')
report_path = os.path.join(RUNNING_PATH,'..',config['production']['prod_deployment_path'])

prediction_model = None

def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """

    parser = ArgumentParser(prog="app",
                            description="API Interface")

    parser.add_argument("-a",
        "--address", 
        type= str,
        help="Listening Address",
        default="0.0.0.0",
        required=False
    )

    parser.add_argument("-p",
        "--port", 
        type=str,
        help="Listening Port",
        default="8000",
        required=False
    )

    return parser.parse_args()


# Welcome Endpoint
@app.route("/")
def greetings():        
    #welcoming message
    return 'Welcome the model API'

# Prediction Endpoint
@app.route("/prediction", methods=['GET','OPTIONS'])
def predict():        
    #call the prediction function you created in Step 3
    if request.method == 'GET':
        file = os.path.join(dataset_csv_path, request.args.get('filename'))
        return {'predictions': str(model_predictions(model_file,file,db_file,LOGGER_=LOGGER))}

# Scoring Endpoint
@app.route("/scoring", methods=['GET','OPTIONS'])
def get_score():        
    #check the score of the deployed model
    data_test_file = os.path.join(dataset_csv_path, 'finaldata.csv')
    return {'F1 score': score_model(data_test_file, model_file, db_file, LOGGER_=LOGGER)}

# Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET','OPTIONS'])
def get_stats():        
    #check means, medians, and modes for each column
    summary = dataframe_summary(db_file, LOGGER_=LOGGER)
    # return summary
    summary_dict = {'key statistics': {c:{'mean':summary[i],
                                  'median':summary[i+4],
                                  'std':summary[i+8]} 
    for c,i in zip(['lastmonth_activity','lastyear_activity','number_of_employees'], range(3))
                                }
            }
    return summary_dict

#######################Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET','OPTIONS'])
def get_diagnostics():        
    #check timing and percent NA values
    missing_data_rep = missing_data(db_file, LOGGER_=LOGGER)
    timing = execution_time()
    dependency_check = outdated_packages_list()
    return {'execution time': {step:duration 
                for step, duration in zip(['ingestion step','training step'],
                                            timing)}, 
            'missing data': {col:pct 
                for col, pct in zip(['lastmonth_activity',
                                    'lastyear_activity',
                                    'number_of_employees',
                                    'exited'], missing_data_rep)},
            'dependency check':[{'Module':row[0], 
                                'Version':row[1][0], 
                                'Vlatest':row[1][1]} 
                                for row in dependency_check.iterrows()]
            }


@app.route('/download', methods=['GET', 'OPTIONS'])
def download():
    return send_from_directory(directory=report_path, path='report.pdf')


def main(args):
    """
    Run the main function

    args: command line arguments
    """

    global LOGGER

    LOGGER.info("Running Flask Server")
    app.run(host=args.address, port=args.port, debug=True, threaded=True)


if __name__ == '__main__':

    computer_name = platform.node()
    SCRIPT_NAME = "app"
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
