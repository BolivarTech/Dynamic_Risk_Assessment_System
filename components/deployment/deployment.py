"""
Model Deployment

MLFlow model deployment step

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
import shutil

# Get the running script path
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

    parser = ArgumentParser(prog="deployment",
                            description="Model Deployment")
    parser.add_argument("-m",
        "--model_path", 
        type=str,
        help="Model Files Path",
        default=os.path.join(RUNNING_PATH,'../../practicemodels'),
        required=False
    )
    parser.add_argument("-i",
        "--ingested_files", 
        type=str,
        help="Ingested files record",
        default=os.path.join(RUNNING_PATH,'../../ingesteddata/ingestedfiles.txt'),
        required=False
    )
    parser.add_argument("-d",
        "--deploy_path", 
        type=str,
        help="Production Deployment Path ",
        default=os.path.join(RUNNING_PATH,'../../production_deployment'),
        required=False
    )

    return parser.parse_args()


def main(args):
    """
    Run the main function

    args: command line arguments
    """

    global LOGGER

    # get files on model_path
    files = os.listdir(args.model_path)
    files = [os.path.join(args.model_path, f) for f in os.listdir(args.model_path)
            if os.path.isfile(os.path.join(args.model_path, f))
            and (f.endswith(".txt") or f.endswith(".pkl") )  ]
    files.append(args.ingested_files)
    # Clean deploy path
    if os.path.exists(args.deploy_path):
        shutil.rmtree(args.deploy_path)
    os.mkdir(args.deploy_path)
    # Move files ot deploy path
    for file in files:
        try:
            shutil.move(file, args.deploy_path)
        except Exception as err:
                LOGGER.error(f"Coping File {file} error (001)\n{err}")
        else:
            LOGGER.info(f"File {file} copied")


if __name__ == '__main__':

    computer_name = platform.node()
    SCRIPT_NAME = "deployment"
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
    LOGGER.debug("Running... (002)")
    args = build_argparser()
    main(args)
    LOGGER.debug("Finished. (003)")
