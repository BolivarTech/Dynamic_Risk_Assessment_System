"""
ML ingestion

Process raw data into the pipeline

By: Julian Bolivar
Version: 1.0.0
Date:  2023/06/12
Revision 1.0.0 ( 2023/06/12 ): Initial Release
"""

# Main System Imports
from argparse import ArgumentParser
import logging as log
import logging.handlers
import sys
import os
import platform
from datetime import datetime as dt

# Data Science Imports
import pandas as pd

# Data Base Imports
import sqlite3 as db


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

    parser.add_argument(
        "-i",
        "--input_path",
        type=str,
        help="Path from where the data is ingested into the pipeline",
        default='../../practicedata',
        required=True)
    parser.add_argument("-o",
                        "--output_file",
                        type=str,
                        help="File where the prepocessed data is stored",
                        default='./finaldata.sqlite',
                        required=False
                        )
    parser.add_argument("-r",
                        "--record_file",
                        type=str,
                        help="File where the prepocessed data is stored",
                        default='./ingestedfiles.txt',
                        required=False
                        )

    return parser.parse_args()


def read_csv(filename):
    """
    read a csv file into a pandas' data frame

    input: cvs filenames to read

    output: pandas'dataframe
    """
    return pd.read_csv(filename)


def merge_multiple_dataframe(files_to_ingest, args):
    """
    Merge multiple csv datasets into one master file

    inputs:
        files_to_ingest: List with the files to be ingested
        args: command line arguments
    output: Master dataset and list of ingested files saved to disk
    """

    global LOGGER

    # master dataset placeholder
    finaldata = pd.DataFrame()
    # ingested files placeholder
    ingestedfiles = []

    # compile datasets together and store ingested file names
    for file in files_to_ingest:
        temp = read_csv(file)
        finaldata = pd.concat([finaldata, temp], axis=0)
        ingestedfiles.append(file)

    # drop duplicates
    org_length = len(finaldata)
    finaldata.drop_duplicates(inplace=True)
    after_length = len(finaldata)

    LOGGER.info(f"Duplicated Removed: {org_length-after_length} (001)")

    #connect to a database, creating it if it doesn't exist 
    conn = db.connect(args.output_file)
    LOGGER.info(f"Cleaned Data File: {args.output_file} (002)")

    if conn is not None:
        try: 
            # write dataset to the database file
            finaldata.to_sql("ingested_data", conn, if_exists="replace", index=False)
            LOGGER.info(f"Ingested Data created into {args.output_file} (002)")
            # get current time
            now = dt.now().strftime("%Y-%m-%d %H:%M:%S")
            # Create Ingested Files Data Frame
            ingestedfiles_df = pd.DataFrame(list(zip([now]*len(ingestedfiles),ingestedfiles)))
            ingestedfiles_df.columns = ['date', 'file']
            # Save ingested files to database
            ingestedfiles_df.to_sql("ingested_files", conn, if_exists="replace", index=False)
            LOGGER.info(f"Ingested Files created into {args.output_file} (002)")
    
        except ValueError:
            # if exception occour Rollback
            conn.rollback()
            LOGGER.error(f"Can't create table 'ingested_data' in {args.output_file} (002)")
        else:
            # commit the transaction
            conn.commit()
            LOGGER.debug(f"Transactions commited (001)")
        finally:
            # close out the connection
            conn.close()
            LOGGER.debug(f"Connection Closed (001)")
    else:
        LOGGER.error(f"Can't connect with {args.output_file} (002)")

    # save data on plain csv file
    file_name = os.path.basename(args.output_file).split('.', 1)[0]
    file_name = file_name + '.csv' 
    file_path = os.path.dirname(args.output_file)
    file_csv = os.path.join(file_path, file_name)
    finaldata.to_csv(file_csv, index=False)

    # save ingested files on plain text file
    with open(args.record_file, 'w') as file:
        file.write(str(ingestedfiles))

    files_list = "\n".join(ingestedfiles)
    LOGGER.info(f"Ingested Files: (003)\n {files_list}")


def main(args):
    """
    Run the main function

    args: command line arguments
    """

    global LOGGER

    # get files on input_path
    files = os.listdir(args.input_path)
    # Filtering only the .csv files.
    files = [os.path.join(args.input_path, f) for f in files
             if os.path.isfile(os.path.join(args.input_path, f))
             and f.endswith(".csv")]
    files_list = "\n".join(files)
    LOGGER.info(f"Files Found In: {args.input_path} (004)\n{files_list}")
    # Ingest the files
    merge_multiple_dataframe(files, args)


if __name__ == '__main__':

    computer_name = platform.node()
    SCRIPT_NAME = "ingestion"
    loggPath = os.path.join(".", "log")
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
    logFormatter = log.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s: %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S')
    LOGHANDLER.setFormatter(logFormatter)
    # Add handler to logger
    if 'LOGHANDLER' in globals():
        LOGGER.addHandler(LOGHANDLER)
    else:
        LOGGER.debug("logHandler NOT defined (005)")
    # Set Logger Lever
    LOGGER.setLevel(LOGLEVEL_)
    # Start Running
    LOGGER.debug("Running... (006)")
    args = build_argparser()
    main(args)
    LOGGER.debug("Finished. (007)")
