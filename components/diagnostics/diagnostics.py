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

    parser = ArgumentParser(prog="diagnostics",
                            description="Model and data diagnostics")


    parser.add_argument("-p",
        "--deploy_path", 
        type=str,
        help="Path to deploymed model",
        default=os.path.join(RUNNING_PATH,'../../production_deployment'),
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
        default=os.path.join(RUNNING_PATH,'../../db'),
        required=False
    )


    return parser.parse_args()


def main(args):
    """
    Run the main function

    args: command line arguments
    """

    global LOGGER

    ######################
    # YOUR CODE HERE     #
    ######################


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
