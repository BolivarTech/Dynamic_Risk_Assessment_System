"""
ML ingestion

Process raw data into the pipeline

By: Julian Bolivar
Version: 1.0.0
Date:  ## Set Release Date Here  ##
Revision 1.0.0 ( ## Set Release Date ## ): Initial Release
"""

# Main System Imports
from argparse import ArgumentParser
import logging as log
import logging.handlers
import sys
import os
import platform

# Main Logger
LOGHANDLER = None
LOGGER = None
LOGLEVEL_ = logging.INFO


def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """

    parser = ArgumentParser(prog="ingestion",
                            description="ML ingestion")

    parser.add_argument("-i",
        "--input_path", 
        type=str,
        help="Path from where the data is ingested into the pipeline",
        default='../../practicedata',
        required=True
    )
    parser.add_argument("-o",
        "--output_file", 
        type=str,
        help="File where the prepocessed data is stored",
        default='./finaldata.csv',
        required=False
    )

    return parser.parse_args()


def main(args):
    """
    Run the main function

    args: command line arguments
    """

    global LOGGER

    files = os.listdir(args.input_path)
    #Filtering only the .csv files.
    files = [f for f in files if os.path.isfile(os.path.join(args.input_path,f)) and f.endswith(".csv")] 
    #print(*files, sep="\n")
    print(files)
    LOGGER.info(f"Files In: {args.input_path} (001)\n{files}")


if __name__ == '__main__':

    computer_name = platform.node()
    SCRIPT_NAME = "ingestion"
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
