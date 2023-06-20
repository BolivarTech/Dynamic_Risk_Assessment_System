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

    parser = ArgumentParser(prog="fullprocess",
                            description="ML monitoring ")


    parser.add_argument(
        "--parameter1", 
        type=## INSERT TYPE HERE: str, float or int,
        help=## INSERT DESCRIPTION HERE,
        default=## INSERT DEFAULT VALUE HERE OR os.path.join(RUNNING_PATH,'## RELATIVE PATH ##'),
        required=True
    )

    parser.add_argument(
        "--parameter2", 
        type=## INSERT TYPE HERE: str, float or int,
        help=## INSERT DESCRIPTION HERE,
        default=## INSERT DEFAULT VALUE HERE OR os.path.join(RUNNING_PATH,'## RELATIVE PATH ##'),
        required=True
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
    main(args)
    LOGGER.debug("Finished. (001)")
