import time as time_true
import pathlib
import pandas as pd
import json
import logging

import datetime
from datetime import datetime
from datetime import timedelta
from datetime import timezone
import pytz

from typing import List
from typing import Dict
from typing import Union


import yfinance as yf

class PyRobot():

    def __init__(self, client_id: str, \
                 redirect_uri: str, \
                 lgfile = None
                 ) -> None:
        """Initalizes a new instance of the robot and logs into the API platform specified.

        Arguments:
        ----
        client_id {str} -- The Consumer ID assigned to you during the App registration.
            This can be found at the app registration portal.

        redirect_uri {str} -- This is the redirect URL

        Keyword Arguments:
        ----
        credentials_path {str} -- The path to the session state file used to prevent a full
            OAuth workflow. (default: {None})

        """

        # Set the attributes

        self.client_id = client_id
        self.redirect_uri = redirect_uri
        self.portfolio = {}
        self.bot_portfolio = {}
        self.old_responses = []
        self.historical_prices = {}
        self.logfiler = lgfile
        self.historical_prices_df = pd.DataFrame()

        # Trading stuff
        self.signals = []
        self.call_options = []
        self.put_options = []
        self.filled_orders = []



    def milliseconds_since_epoch(self, dt_object: datetime) -> int:
        """converts a datetime object to milliseconds since 1970, as an integer

        Arguments:
        ----------
        dt_object {datetime.datetime} -- Python datetime object.

        Returns:
        --------
        [int] -- The timestamp in milliseconds since epoch.
        """

        return int(dt_object.timestamp() * 1000)

    def datetime_from_milliseconds_since_epoch(self, ms_since_epoch: int, timezone: timezone = None) -> datetime :
        """Converts milliseconds since epoch to a datetime object.

        Arguments:
        ----------
        ms_since_epoch {int} -- Number of milliseconds since epoch.

        Keyword Arguments:
        --------
        timezone {datetime.timezone} -- The timezone of the new datetime object. (default: {None})

        Returns:
        --------
        datetime.datetime -- A python datetime object.
        """

        return datetime.fromtimestamp((ms_since_epoch / 1000), tz=timezone)


    def grab_historical_prices(self, start: str = '', end: str = '', bar_string: str = '',
                               period_string: str = '', symbols: List[str] = None) -> pd.DataFrame:
        """Grabs the historical prices for all the tickers

        Overview:
        ----
        Any of the historical price data returned will include extended hours
        price data by default.

        Arguments:
        ----
        start {string} -- Defines the start date for the historical prices.

        end {string} -- Defines the end date for the historical prices.
        """

        # This is the only endpoint used in the Yahoo library
        yahoo_prices_df = yf.download(tickers=symbols[0], interval=bar_string, start=start, end=end, auto_adjust = True, threads=False)
        #yahoo_prices_df = yahoo_prices_df['High'] + yahoo_prices_df['Low'] + yahoo_prices_df['Open'] + yahoo_prices_df['Close'] + yahoo_prices_df['Volume'] # Renaming columns to maintain compatibility with previous versions of the bot
        self.historical_prices_df = yahoo_prices_df.rename(columns={'High':'high', 'Low' :'low', 'Open': 'open', 'Close' : 'close', 'Volume':'volume'})
        self.historical_prices_df.insert(0,'symbol',symbols[0])
        #self.historical_prices_df = self.historical_prices_df.set_index(keys=['symbol'])
        #print(self.historical_prices_df.head())
        self.stock_frame = self.historical_prices_df
        return self.historical_prices_df




    
