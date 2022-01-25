import platform
import logging
import glob
import os
import pandas as pd
from os.path import exists

from datetime import datetime
import pytz
from pytz import timezone
from configparser import ConfigParser
from robot import PyRobot

def setup_func(logger_hndl=None):
    # Get credentials

    import_credentials(log_hndl=logger_hndl)    # Remnants of another bot

    # Initalize the robot with my credentials.
    trading_robot = PyRobot(
        client_id='Test_ID',                     # Not required
        redirect_uri='Test_URI',                 # Not Required
        lgfile = logger_hndl
    )

    # J. Jones - setting bots default timezone to EST.
    est_tz = pytz.timezone('US/Eastern')
    dt = datetime.now(est_tz).strftime("%Y_%m_%d-%I%M%S_%p")
    logmsg = "Session created at " + dt + " EST"
    logger_hndl.info(logmsg)
    logger_hndl.info("Downloading Data")


    logmsg = '='*80
    logger_hndl.info(logmsg)

    return trading_robot


def import_credentials(log_hndl=None):
    system = platform.system()
    config = ConfigParser()
    currWorkingDirectory = os.getcwd()
    log_hndl.info("Working from default directory %s ", currWorkingDirectory)



    return 'client_id', 'http://localhost:5000'
