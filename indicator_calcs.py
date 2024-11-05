import numpy as np
import pandas as pd
import operator
import math
import re
import json


from typing import Any
from typing import Dict
from typing import Union
from typing import Optional
from typing import Tuple
import datetime
from datetime import timezone
from datetime import timedelta
import pytz



pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.set_option('display.max_columns', None)



class Indicators():
    """
    Represents an Indicator Object which can be used
    to easily add technical indicators to a StockFrame.
    """

    def __init__(self, input_price_data_frame: pd.DataFrame, lgfile=None) -> None:
        """Initalizes the Indicator Client.

        Arguments:
        ----
        price_data_frame  -- The price data frame which is used to add indicators to.
            At a minimum this data frame must have the following columns: `['timestamp','close','open','high','low']`.

        Usage:
        ----
            >>> historical_prices_df = trading_robot.grab_historical_prices(
                start=start_date,
                end=end_date,
                bar_size=1,
                bar_type='minute'
            )
            >>> price_data_frame = pd.DataFrame(data=historical_prices)
            >>> indicator_client = Indicators(price_data_frame=price_data_frame)
            >>> indicator_client.price_data_frame
        """

        self._indicator_signals = {}
        self.stock_data = input_price_data_frame
        self.indicator_signal_list = []
        self.symbol = ''

        # TODO: use Alex's add_rows() function instead of updating whole dataframe
        self._frame = self.stock_data

        self._indicators_comp_key = []
        self._indicators_key = []
        self.logfiler = lgfile


    def get_indicator_signal(self, indicator: str = None) -> Dict:
        """Return the raw Pandas Dataframe Object.

        Arguments:
        ----
        indicator {Optional[str]} -- The indicator key, for example `ema` or `sma`.

        Returns:
        ----
        {dict} -- Either all of the indicators or the specified indicator.
        """

        if indicator and indicator in self._indicator_signals:
            return self._indicator_signals[indicator]
        else:
            return self._indicator_signals

    def set_indicator_signal(self, indicator: str, buy: float, sell: float, condition_buy: Any, condition_sell: Any,
                             buy_max: float = None, sell_max: float = None, condition_buy_max: Any = None,
                             condition_sell_max: Any = None) -> None:
        """Used to set an indicator where one indicator crosses above or below a certain numerical threshold.

        Arguments:
        ----
        indicator {str} -- The indicator key, for example `ema` or `sma`.

        buy {float} -- The buy signal threshold for the indicator.

        sell {float} -- The sell signal threshold for the indicator.

        condition_buy {str} -- The operator which is used to evaluate the `buy` condition. For example, `">"` would
            represent greater than or from the `operator` module it would represent `operator.gt`.

        condition_sell {str} -- The operator which is used to evaluate the `sell` condition. For example, `">"` would
            represent greater than or from the `operator` module it would represent `operator.gt`.

        buy_max {float} -- If the buy threshold has a maximum value that needs to be set, then set the `buy_max` threshold.
            This means if the signal exceeds this amount it WILL NOT PURCHASE THE INSTRUMENT. (defaults to None).

        sell_max {float} -- If the sell threshold has a maximum value that needs to be set, then set the `buy_max` threshold.
            This means if the signal exceeds this amount it WILL NOT SELL THE INSTRUMENT. (defaults to None).

        condition_buy_max {str} -- The operator which is used to evaluate the `buy_max` condition. For example, `">"` would
            represent greater than or from the `operator` module it would represent `operator.gt`. (defaults to None).

        condition_sell_max {str} -- The operator which is used to evaluate the `sell_max` condition. For example, `">"` would
            represent greater than or from the `operator` module it would represent `operator.gt`. (defaults to None).
        """

        # Add the key if it doesn't exist.
        if indicator not in self._indicator_signals:
            self._indicator_signals[indicator] = {}
            self._indicators_key.append(indicator)

            # Add the signals.
        self._indicator_signals[indicator]['buy'] = buy
        self._indicator_signals[indicator]['sell'] = sell
        self._indicator_signals[indicator]['buy_operator'] = condition_buy
        self._indicator_signals[indicator]['sell_operator'] = condition_sell

        # Add the max signals
        self._indicator_signals[indicator]['buy_max'] = buy_max
        self._indicator_signals[indicator]['sell_max'] = sell_max
        self._indicator_signals[indicator]['buy_operator_max'] = condition_buy_max
        self._indicator_signals[indicator]['sell_operator_max'] = condition_sell_max

    def set_indicator_signal_compare(self, indicator_1: str, indicator_2: str, condition_buy: Any,
                                     condition_sell: Any) -> None:
        """Used to set an indicator where one indicator is compared to another indicator.

        Overview:
        ----
        Some trading strategies depend on comparing one indicator to another indicator.
        For example, the Simple Moving Average crossing above or below the Exponential
        Moving Average. This will be used to help build those strategies that depend
        on this type of structure.

        Arguments:
        ----
        indicator_1 {str} -- The first indicator key, for example `ema` or `sma`.

        indicator_2 {str} -- The second indicator key, this is the indicator we will compare to. For example,
            is the `sma` greater than the `ema`.

        condition_buy {str} -- The operator which is used to evaluate the `buy` condition. For example, `">"` would
            represent greater than or from the `operator` module it would represent `operator.gt`.

        condition_sell {str} -- The operator which is used to evaluate the `sell` condition. For example, `">"` would
            represent greater than or from the `operator` module it would represent `operator.gt`.
        """

        # Define the key.
        key = "{ind_1}_comp_{ind_2}".format(
            ind_1=indicator_1,
            ind_2=indicator_2
        )

        # Add the key if it doesn't exist.
        if key not in self._indicator_signals:
            self._indicator_signals[key] = {}
            self._indicators_comp_key.append(key)

            # Grab the dictionary.
        indicator_dict = self._indicator_signals[key]

        # Add the signals.
        indicator_dict['type'] = 'comparison'
        indicator_dict['indicator_1'] = indicator_1
        indicator_dict['indicator_2'] = indicator_2
        indicator_dict['buy_operator'] = condition_buy
        indicator_dict['sell_operator'] = condition_sell

    @property
    def price_data_frame(self) -> pd.DataFrame:
        """Return the raw Pandas Dataframe Object.

        Returns:
        ----
        {pd.DataFrame} -- A multi-index data frame.
        """

        return self._frame

    @price_data_frame.setter
    def price_data_frame(self, price_data_frame: pd.DataFrame) -> None:
        """Sets the price data frame.

        Arguments:
        ----
        price_data_frame {pd.DataFrame} -- A multi-index data frame.
        """

        self._frame = price_data_frame

    @property
    def is_multi_index(self) -> bool:
        """Specifies whether the data frame is a multi-index dataframe.

        Returns:
        ----
        {bool} -- `True` if the data frame is a `pd.MultiIndex` object. `False` otherwise.
        """

        if isinstance(self._frame.index, pd.MultiIndex):
            return True
        else:
            return False

    def change_in_price(self, column_name: str = 'change_in_price') -> pd.DataFrame:
        """Calculates the Change in Price.

        Returns:
        ----
        {pd.DataFrame} -- A data frame with the Change in Price included.
        """

        self._frame[column_name] = self._frame['close'].transform(
            lambda x: x.diff()
        )

        return self._frame


    def ema(self, period: int, column_name: str = 'ema') -> pd.DataFrame:
        """Calculates the Exponential Moving Average (EMA).

        Arguments:
        ----
        period {int} -- The number of periods to use when calculating the EMA.

        Returns:
        ----
        {pd.DataFrame} -- A Pandas data frame with the SMA indicator included.

        Usage:
        ----
            >>> historical_prices_df = trading_robot.grab_historical_prices(
                start=start_date,
                end=end_date,
                bar_size=1,
                bar_type='minute'
            )
            >>> price_data_frame = pd.DataFrame(data=historical_prices)
            >>> indicator_client = Indicators(price_data_frame=price_data_frame)
            >>> indicator_client.ewm(period=100)
        """


        # print(self.price_data_frame)
        # Add the ema
        self._frame[column_name + '_' + str(period), self.symbol] = self._frame['close', self.symbol].transform(
            lambda x: x.ewm(span=period, min_periods=period, adjust=False, ignore_na = False).mean()
        )


        return self._frame


    def per_of_change(self) -> pd.DataFrame:

        # calculate per of change
        per_of_change = []
        count = 0

        while count < len(self._frame):
            if count == 0:
                per_of_change.append(0)
                prev = next = self._frame['close', self.symbol].iloc[count]
            else:
                prev = next
                next = self._frame['close',self.symbol].iloc[count]
                per_of_change.append(round(((next - prev) / prev) * 100, 3))
            count += 1

        self._frame['per_of_change', self.symbol] = pd.Series(per_of_change).values

        return self._frame

    # Comparison of EMA9, EMA50 and EMA200
    #
    def ema_comparisions(self):

        resulting_comparison = []
        count = 0

        while count < len(self._frame):
            if count == 0:
                resulting_comparison.append(False)
            elif math.isnan(self._frame["ema_9", self.symbol].iloc[count]) or math.isnan(self._frame["ema_50", self.symbol].iloc[count]) or math.isnan(self._frame["ema_200", self.symbol].iloc[count]) :
                resulting_comparison.append(False)
            elif (self._frame["ema_9", self.symbol].iloc[count] > self._frame["ema_50", self.symbol].iloc[count]) and (self._frame["ema_50", self.symbol].iloc[count] > self._frame["ema_200", self.symbol].iloc[count]) :
                resulting_comparison.append(True)
            else:
                resulting_comparison.append(False)
            count += 1

        self._frame['EMA9_EMA50_EMA200', self.symbol] = pd.Series(resulting_comparison).values

        return self._frame


    def filter_non_numeric(self, input_str=str) -> str :
        numeric_str = re.sub("[^\d\.]", "", input_str)
        return numeric_str

    def buy_condition(self):

        signal_list = []
        Open_Position = []
        Purchase_Sell = []
        signal_list.append('')
        Open_Position.append(0)
        Purchase_Sell.append(0)

        # Generates signals column called buy_condition
        for i in range(1,len(self._frame)):
            if (self._frame['EMA9_EMA50_EMA200', self.symbol].iloc[i]) and (Open_Position[i-1] <= 0) and (self._frame['14_Day_Stoch_RSI', self.symbol].iloc[i] < .3) and (self._frame['RSI_K_Line', self.symbol].iloc[i] >  self._frame['RSI_D_Line', self.symbol].iloc[i]) :
                signal_list.append('Buy')
                Open_Position.append(1)
                Purchase_Sell.append(self._frame['close', self.symbol].iloc[i-1] * -1)
            elif (self._frame['RSI_K_Line', self.symbol].iloc[i] < self._frame['RSI_D_Line', self.symbol].iloc[i]) and (Open_Position[i-1] > 0) and (self._frame['14_Day_Stoch_RSI', self.symbol].iloc[i] > .9):
                signal_list.append('Sell')
                Open_Position.append(0)
                Purchase_Sell.append(self._frame['close', self.symbol].iloc[i-1])
            elif (i == (len(self._frame)-1)) and (Open_Position[i-1] > 0) :
                signal_list.append('Sell')
                Open_Position.append(0)
                Purchase_Sell.append(self._frame['close', self.symbol].iloc[i-1])
            else:
                signal_list.append('')
                Open_Position.append(Open_Position[i-1])
                Purchase_Sell.append(0)
        self._frame['Purchase_Sell', self.symbol] = pd.Series(Purchase_Sell).values
        self._frame["buy_sell_condition", self.symbol] = pd.Series(signal_list).values
        self._frame['Open_Position', self.symbol] = pd.Series(Open_Position).values
        self.stock_data = self._frame

        Running_PnL = []
        Running_PnL.append(0)
        for i in range(1, len(self._frame)):
            running_total=0
            for x in range(i,0,-1) :
                running_total += self._frame['Purchase_Sell', self.symbol].iloc[x]
            Running_PnL.append(running_total)
        self._frame['Running_PnL', self.symbol] = pd.Series(Running_PnL).values
        return self._frame


    def rate_of_change(self, period: int = 1, column_name: str = 'rate_of_change') -> pd.DataFrame:
        """Calculates the Rate of Change (ROC).

        Arguments:
        ----
        period {int} -- The number of periods to use when calculating
            the ROC. (default: {1})

        Returns:
        ----
        {pd.DataFrame} -- A Pandas data frame with the ROC indicator included.

        Usage:
        ----
            >>> historical_prices_df = trading_robot.grab_historical_prices(
                start=start_date,
                end=end_date,
                bar_size=1,
                bar_type='minute'
            )
            >>> price_data_frame = pd.DataFrame(data=historical_prices)
            >>> indicator_client = Indicators(price_data_frame=price_data_frame)
            >>> indicator_client.rate_of_change()
        """



        self._frame[column_name, self.symbol] = self._frame['close', self.symbol].transform(
            lambda x: x.pct_change(periods=period)
        )

        return self._frame

    def gain_loss(self, column_name: str = None) -> pd.DataFrame:
        close_prices = self._frame['close', self.symbol]
        gain_column = []
        loss_column = []
        row_counter = 0
        for price in close_prices :
            if row_counter > 0 :
                price_diff = close_prices.iloc[row_counter] - close_prices.iloc[row_counter-1]
                if price_diff < 0 :
                    loss_column.append(price_diff*(-1.0))
                    gain_column.append(0.0)
                elif price_diff > 0 :
                    loss_column.append(0)
                    gain_column.append(price_diff)
                else:
                    loss_column.append(0)
                    gain_column.append(0.0)
                row_counter += 1
            else :
                loss_column.append(0)
                gain_column.append(0.0)
                row_counter += 1

        self._frame['Gain', self.symbol] = gain_column
        self._frame['Loss', self.symbol] = loss_column

        return self._frame

    def rsi(self, column_name: str = None) -> pd.DataFrame:
        Avg_Loss_Col = self._frame['AvgLoss', self.symbol]
        RS_Col = self._frame['RS', self.symbol]
        rsi = []
        row_counter = 0
        for loss_item in Avg_Loss_Col :
            if loss_item == 0 :
                rsi.append(100.0)
            elif math.isnan(loss_item) :
                rsi.append(float('nan'))
            else:
                rsi.append(100.0-(100/(1+RS_Col.iloc[row_counter])))
            row_counter += 1


        self._frame[column_name,self.symbol] = rsi

        return self._frame

    def rel_strength(self) -> pd.DataFrame :
        avg_gain_column = self._frame['AvgGain', self.symbol]
        avg_loss_column = self._frame['AvgLoss', self.symbol]
        RS_Column = avg_gain_column /  avg_loss_column

        self._frame['RS', self.symbol] = RS_Column

        return self._frame

    def sma(self, period: int, result_column_name: str = None, input_column_name: str = None) -> pd.DataFrame:
    #    Arguments:
    #    ----
    #    period {int} -- The number of periods to use when calculating the SMA.#

    #    Returns:
    #    ----
    #    {pd.DataFrame} -- A Pandas data frame with the SMA indicator included.


        self._frame[result_column_name, self.symbol] = self._frame[input_column_name, self.symbol].transform(
            lambda x: x.rolling(window=period).mean()
        )

        return self._frame

    def rsi_max(self, period: int, result_column_name: str = None, input_column_name: str = None) -> pd.DataFrame:
    #    Arguments:
    #    ----
    #    period {int} -- The number of periods to use when calculating the SMA.#

    #    Returns:
    #    ----
    #    {pd.DataFrame} -- A Pandas data frame with the SMA indicator included.


        self._frame[result_column_name, self.symbol] = self._frame[input_column_name, self.symbol].transform(
            lambda x: x.rolling(window=period).max()
        )

        return self._frame

    def rsi_min(self, period: int, result_column_name: str = None, input_column_name: str = None) -> pd.DataFrame:
    #    Arguments:
    #    ----
    #    period {int} -- The number of periods to use when calculating the SMA.#

    #    Returns:
    #    ----
    #    {pd.DataFrame} -- A Pandas data frame with the SMA indicator included.

        self._frame[result_column_name, self.symbol] = self._frame[input_column_name, self.symbol].transform(
            lambda x: x.rolling(window=period).min()
        )

        return self._frame

    def stoch_rsi(self, column_name: str = None) -> pd.DataFrame:
        RSI_Col = self._frame['RSI', self.symbol]
        RSI_High_Col = self._frame['RSI_Highest_High', self.symbol]
        RSI_Low_Col = self._frame['RSI_Lowest_Low', self.symbol]
        stoch_rsi=[]
        row_counter = 0
        for RSI_item in RSI_Col :
            numerator = RSI_item - RSI_Low_Col.iloc[row_counter]
            denominator = RSI_High_Col.iloc[row_counter]-RSI_Low_Col.iloc[row_counter]
            stoch_rsi.append(numerator / denominator)
            row_counter += 1
        self._frame[column_name, self.symbol] = stoch_rsi
        return self._frame


    def option_volume_for_candle(self, period: int = 1, column_name: str = 'call or put total volume') -> pd.DataFrame:

        return self._frame



    def bollinger_bands(self, period: int = 20, column_name: str = 'bollinger_bands') -> pd.DataFrame:
        """Calculates the Bollinger Bands.

        Arguments:
        ----
        period {int} -- The number of periods to use when calculating
            the Bollinger Bands. (default: {20})

        Returns:
        ----
        {pd.DataFrame} -- A Pandas data frame with the Lower and Upper band
            indicator included.

        Usage:
        ----
            >>> historical_prices_df = trading_robot.grab_historical_prices(
                start=start_date,
                end=end_date,
                bar_size=1,
                bar_type='minute'
            )
            >>> price_data_frame = pd.DataFrame(data=historical_prices)
            >>> indicator_client = Indicators(price_data_frame=price_data_frame)
            >>> indicator_client.bollinger_bands()
        """

        # Define the Moving Avg.
        self._frame['moving_avg'] = self._frame['close'].transform(
            lambda x: x.rolling(window=period).mean()
        )

        # Define Moving Std.
        self._frame['moving_std'] = self._frame['close'].transform(
            lambda x: x.rolling(window=period).std()
        )

        # Define the Upper Band.
        self._frame['band_upper'] = 4 * (self._frame['moving_std'] / self._frame['moving_avg'])

        # Define the lower band
        self._frame['band_lower'] = (
                (self._frame['close'] - self._frame['moving_avg']) +
                (2 * self._frame['moving_std']) /
                (4 * self._frame['moving_std'])
        )

        # Clean up before sending back.
        self._frame.drop(
            labels=['moving_avg', 'moving_std'],
            axis=1,
            inplace=True
        )

        return self._frame

    def average_true_range(self, period: int = 14, column_name: str = 'average_true_range') -> pd.DataFrame:
        """Calculates the Average True Range (ATR).

        Arguments:
        ----
        period {int} -- The number of periods to use when calculating
            the ATR. (default: {14})

        Returns:
        ----
        {pd.DataFrame} -- A Pandas data frame with the ATR included.

        Usage:
        ----
            >>> historical_prices_df = trading_robot.grab_historical_prices(
                start=start_date,
                end=end_date,
                bar_size=1,
                bar_type='minute'
            )
            >>> price_data_frame = pd.DataFrame(data=historical_prices)
            >>> indicator_client = Indicators(price_data_frame=price_data_frame)
            >>> indicator_client.average_true_range()
        """

        # Calculate the different parts of True Range.
        self._frame['true_range_0'] = abs(self._frame['high'] - self._frame['low'])
        self._frame['true_range_1'] = abs(self._frame['high'] - self._frame['close'].shift())
        self._frame['true_range_2'] = abs(self._frame['low'] - self._frame['close'].shift())

        # Grab the Max.
        self._frame['true_range'] = self._frame[['true_range_0', 'true_range_1', 'true_range_2']].max(axis=1)

        # Calculate the Average True Range.
        self._frame['average_true_range'] = self._frame['true_range'].transform(
            lambda x: x.ewm(span=period, min_periods=period).mean()
        )

        # Clean up before sending back.
        self._frame.drop(
            labels=['true_range_0', 'true_range_1', 'true_range_2', 'true_range'],
            axis=1,
            inplace=True
        )

        return self._frame

    def stochastic_oscillator(self, column_name: str = 'stochastic_oscillator') -> pd.DataFrame:
        """Calculates the Stochastic Oscillator.

        Returns:
        ----
        {pd.DataFrame} -- A Pandas data frame with the Stochastic Oscillator included.

        Usage:
        ----
            >>> historical_prices_df = trading_robot.grab_historical_prices(
                start=start_date,
                end=end_date,
                bar_size=1,
                bar_type='minute'
            )
            >>> price_data_frame = pd.DataFrame(data=historical_prices)
            >>> indicator_client = Indicators(price_data_frame=price_data_frame)
            >>> indicator_client.stochastic_oscillator()
        """

        # Calculate the stochastic_oscillator.
        self._frame['stochastic_oscillator'] = (
                self._frame['close'] - self._frame['low'] /
                self._frame['high'] - self._frame['low']
        )

        return self._frame

    def macd(self, fast_period: int = 12, slow_period: int = 26, column_name: str = 'macd') -> pd.DataFrame:
        """Calculates the Moving Average Convergence Divergence (MACD).

        Arguments:
        ----
        fast_period {int} -- The number of periods to use when calculating
            the fast moving MACD. (default: {12})

        slow_period {int} -- The number of periods to use when calculating
            the slow moving MACD. (default: {26})

        Returns:
        ----
        {pd.DataFrame} -- A Pandas data frame with the MACD included.

        Usage:
        ----
            >>> historical_prices_df = trading_robot.grab_historical_prices(
                start=start_date,
                end=end_date,
                bar_size=1,
                bar_type='minute'
            )
            >>> price_data_frame = pd.DataFrame(data=historical_prices)
            >>> indicator_client = Indicators(price_data_frame=price_data_frame)
            >>> indicator_client.macd(fast_period=12, slow_period=26)
        """

        # Calculate the Fast Moving MACD.
        self._frame['macd_fast'] = self._frame['close'].transform(
            lambda x: x.ewm(span=fast_period, min_periods=fast_period).mean()
        )

        # Calculate the Slow Moving MACD.
        self._frame['macd_slow'] = self._frame['close'].transform(
            lambda x: x.ewm(span=slow_period, min_periods=slow_period).mean()
        )

        # Calculate the difference between the fast and the slow.
        self._frame['macd_diff'] = self._frame['macd_fast'] - self._frame['macd_slow']

        # Calculate the Exponential moving average of the fast.
        self._frame['macd'] = self._frame['macd_diff'].transform(
            lambda x: x.ewm(span=9, min_periods=8).mean()
        )

        return self._frame

    def mass_index(self, period: int = 9, column_name: str = 'mass_index') -> pd.DataFrame:
        """Calculates the Mass Index indicator.

        Arguments:
        ----
        period {int} -- The number of periods to use when calculating
            the mass index. (default: {9})

        Returns:
        ----
        {pd.DataFrame} -- A Pandas data frame with the Mass Index included.

        Usage:
        ----
            >>> historical_prices_df = trading_robot.grab_historical_prices(
                start=start_date,
                end=end_date,
                bar_size=1,
                bar_type='minute'
            )
            >>> price_data_frame = pd.DataFrame(data=historical_prices)
            >>> indicator_client = Indicators(price_data_frame=price_data_frame)
            >>> indicator_client.mass_index(period=9)
        """

        # Calculate the Diff.
        self._frame['diff'] = self._frame['high'] - self._frame['low']

        # Calculate Mass Index 1
        self._frame['mass_index_1'] = self._frame['diff'].transform(
            lambda x: x.ewm(span=period, min_periods=period - 1).mean()
        )

        # Calculate Mass Index 2
        self._frame['mass_index_2'] = self._frame['mass_index_1'].transform(
            lambda x: x.ewm(span=period, min_periods=period - 1).mean()
        )

        # Grab the raw index.
        self._frame['mass_index_raw'] = self._frame['mass_index_1'] / self._frame['mass_index_2']

        # Calculate the Mass Index.
        self._frame['mass_index'] = self._frame['mass_index_raw'].transform(
            lambda x: x.rolling(window=25).sum()
        )

        # Clean up before sending back.
        self._frame.drop(
            labels=['diff', 'mass_index_1', 'mass_index_2', 'mass_index_raw'],
            axis=1,
            inplace=True
        )

        return self._frame

    def force_index(self, period: int, column_name: str = 'force_index') -> pd.DataFrame:
        """Calculates the Force Index.

        Arguments:
        ----
        period {int} -- The number of periods to use when calculating
            the force index.

        Returns:
        ----
        {pd.DataFrame} -- A Pandas data frame with the force index included.

        Usage:
        ----
            >>> historical_prices_df = trading_robot.grab_historical_prices(
                start=start_date,
                end=end_date,
                bar_size=1,
                bar_type='minute'
            )
            >>> price_data_frame = pd.DataFrame(data=historical_prices)
            >>> indicator_client = Indicators(price_data_frame=price_data_frame)
            >>> indicator_client.force_index(period=9)
        """

        # Calculate the Force Index.
        self._frame[column_name] = self._frame['close'].diff(period) * self._frame['volume'].diff(period)

        return self._frame

    def ease_of_movement(self, period: int, column_name: str = 'ease_of_movement') -> pd.DataFrame:
        """Calculates the Ease of Movement.

        Arguments:
        ----
        period {int} -- The number of periods to use when calculating
            the Ease of Movement.

        Returns:
        ----
        {pd.DataFrame} -- A Pandas data frame with the Ease of Movement included.

        Usage:
        ----
            >>> historical_prices_df = trading_robot.grab_historical_prices(
                start=start_date,
                end=end_date,
                bar_size=1,
                bar_type='minute'
            )
            >>> price_data_frame = pd.DataFrame(data=historical_prices)
            >>> indicator_client = Indicators(price_data_frame=price_data_frame)
            >>> indicator_client.ease_of_movement(period=9)
        """

        # Calculate the ease of movement.
        high_plus_low = (self._frame['high'].diff(1) + self._frame['low'].diff(1))
        diff_divi_vol = (self._frame['high'] - self._frame['low']) / (2 * self._frame['volume'])
        self._frame['ease_of_movement_raw'] = high_plus_low * diff_divi_vol

        # Calculate the Rolling Average of the Ease of Movement.
        self._frame['ease_of_movement'] = self._frame['ease_of_movement_raw'].transform(
            lambda x: x.rolling(window=period).mean()
        )

        # Clean up before sending back.
        self._frame.drop(
            labels=['ease_of_movement_raw'],
            axis=1,
            inplace=True
        )

        return self._frame

    def commodity_channel_index(self, period: int, column_name: str = 'commodity_channel_index') -> pd.DataFrame:
        """Calculates the Commodity Channel Index.

        Arguments:
        ----
        period {int} -- The number of periods to use when calculating
            the Commodity Channel Index.

        Returns:
        ----
        {pd.DataFrame} -- A Pandas data frame with the Commodity Channel Index included.

        Usage:
        ----
            >>> historical_prices_df = trading_robot.grab_historical_prices(
                start=start_date,
                end=end_date,
                bar_size=1,
                bar_type='minute'
            )
            >>> price_data_frame = pd.DataFrame(data=historical_prices)
            >>> indicator_client = Indicators(price_data_frame=price_data_frame)
            >>> indicator_client.commodity_channel_index(period=9)
        """

        # Calculate the Typical Price.
        self._frame['typical_price'] = (self._frame['high'] + self._frame['low'] + self._frame['close']) / 3

        # Calculate the Rolling Average of the Typical Price.
        self._frame['typical_price_mean'] = self._frame['pp'].transform(
            lambda x: x.rolling(window=period).mean()
        )

        # Calculate the Rolling Standard Deviation of the Typical Price.
        self._frame['typical_price_std'] = self._frame['pp'].transform(
            lambda x: x.rolling(window=period).std()
        )

        # Calculate the Commodity Channel Index.
        self._frame[column_name] = self._frame['typical_price_mean'] / self._frame['typical_price_std']

        # Clean up before sending back.
        self._frame.drop(
            labels=['typical_price', 'typical_price_mean', 'typical_price_std'],
            axis=1,
            inplace=True
        )

        return self._frame

    def standard_deviation(self, period: int, column_name: str = 'standard_deviation') -> pd.DataFrame:
        """Calculates the Standard Deviation.

        Arguments:
        ----
        period {int} -- The number of periods to use when calculating
            the standard deviation.

        Returns:
        ----
        {pd.DataFrame} -- A Pandas data frame with the Standard Deviation included.

        Usage:
        ----
            >>> historical_prices_df = trading_robot.grab_historical_prices(
                start=start_date,
                end=end_date,
                bar_size=1,
                bar_type='minute'
            )
            >>> price_data_frame = pd.DataFrame(data=historical_prices)
            >>> indicator_client = Indicators(price_data_frame=price_data_frame)
            >>> indicator_client.standard_deviation(period=9)
        """

        # Calculate the Standard Deviation.
        self._frame[column_name] = self._frame['close'].transform(
            lambda x: x.ewm(span=period).std()
        )

        return self._frame

    def chaikin_oscillator(self, period: int, column_name: str = 'chaikin_oscillator') -> pd.DataFrame:
        """Calculates the Chaikin Oscillator.

        Arguments:
        ----
        period {int} -- The number of periods to use when calculating
            the Chaikin Oscillator.

        Returns:
        ----
        {pd.DataFrame} -- A Pandas data frame with the Chaikin Oscillator included.

        Usage:
        ----
            >>> historical_prices_df = trading_robot.grab_historical_prices(
                start=start_date,
                end=end_date,
                bar_size=1,
                bar_type='minute'
            )
            >>> price_data_frame = pd.DataFrame(data=historical_prices)
            >>> indicator_client = Indicators(price_data_frame=price_data_frame)
            >>> indicator_client.chaikin_oscillator(period=9)
        """

        # Calculate the Money Flow Multiplier.
        money_flow_multiplier_top = 2 * (self._frame['close'] - self._frame['high'] - self._frame['low'])
        money_flow_multiplier_bot = (self._frame['high'] - self._frame['low'])

        # Calculate Money Flow Volume
        self._frame['money_flow_volume'] = (money_flow_multiplier_top / money_flow_multiplier_bot) * self._frame[
            'volume']

        # Calculate the 3-Day moving average of the Money Flow Volume.
        self._frame['money_flow_volume_3'] = self._frame['money_flow_volume'].transform(
            lambda x: x.ewm(span=3, min_periods=2).mean()
        )

        # Calculate the 10-Day moving average of the Money Flow Volume.
        self._frame['money_flow_volume_10'] = self._frame['money_flow_volume'].transform(
            lambda x: x.ewm(span=10, min_periods=9).mean()
        )

        # Calculate the Chaikin Oscillator.
        self._frame[column_name] = self._frame['money_flow_volume_3'] - self._frame['money_flow_volume_10']

        # Clean up before sending back.
        self._frame.drop(
            labels=['money_flow_volume_3', 'money_flow_volume_10', 'money_flow_volume'],
            axis=1,
            inplace=True
        )

        return self._frame

    def kst_oscillator(self, r1: int, r2: int, r3: int, r4: int, n1: int, n2: int, n3: int, n4: int,
                       column_name: str = 'kst_oscillator') -> pd.DataFrame:
        """Calculates the Mass Index indicator.

        Arguments:
        ----
        period {int} -- The number of periods to use when calculating
            the mass index. (default: {9})

        Returns:
        ----
        {pd.DataFrame} -- A Pandas data frame with the Mass Index included.

        Usage:
        ----
            >>> historical_prices_df = trading_robot.grab_historical_prices(
                start=start_date,
                end=end_date,
                bar_size=1,
                bar_type='minute'
            )
            >>> price_data_frame = pd.DataFrame(data=historical_prices)
            >>> indicator_client = Indicators(price_data_frame=price_data_frame)
            >>> indicator_client.mass_index(period=9)
        """

        # Calculate the ROC 1.
        self._frame['roc_1'] = self._frame['close'].diff(r1 - 1) / self._frame['close'].shift(r1 - 1)

        # Calculate the ROC 2.
        self._frame['roc_2'] = self._frame['close'].diff(r2 - 1) / self._frame['close'].shift(r2 - 1)

        # Calculate the ROC 3.
        self._frame['roc_3'] = self._frame['close'].diff(r3 - 1) / self._frame['close'].shift(r3 - 1)

        # Calculate the ROC 4.
        self._frame['roc_4'] = self._frame['close'].diff(r4 - 1) / self._frame['close'].shift(r4 - 1)

        # Calculate the Mass Index.
        self._frame['roc_1_n'] = self._frame['roc_1'].transform(
            lambda x: x.rolling(window=n1).sum()
        )

        # Calculate the Mass Index.
        self._frame['roc_2_n'] = self._frame['roc_2'].transform(
            lambda x: x.rolling(window=n2).sum()
        )

        # Calculate the Mass Index.
        self._frame['roc_3_n'] = self._frame['roc_3'].transform(
            lambda x: x.rolling(window=n3).sum()
        )

        # Calculate the Mass Index.
        self._frame['roc_4_n'] = self._frame['roc_4'].transform(
            lambda x: x.rolling(window=n4).sum()
        )

        self._frame[column_name] = 100 * (
                    self._frame['roc_1_n'] + 2 * self._frame['roc_2_n'] + 3 * self._frame['roc_3_n'] + 4 * self._frame[
                'roc_4_n'])
        self._frame[column_name + "_signal"] = self._frame['column_name'].transform(
            lambda x: x.rolling().mean()
        )

        # Clean up before sending back.
        self._frame.drop(
            labels=['roc_1', 'roc_2', 'roc_3', 'roc_4', 'roc_1_n', 'roc_2_n', 'roc_3_n', 'roc_4_n'],
            axis=1,
            inplace=True
        )

        return self._frame


    def refresh(self, symbol) -> pd.DataFrame:
        """Updates the Indicator columns.  It is assumed that the pandas table is already referenced within the object."""

        # First update the groups since, we have new rows.
        self.logfiler.info("Starting Calc Of Indicators")
        self.symbol = symbol
        self.per_of_change()
        self.ema(period=3)
        self.ema(period=9)
        self.ema(period=50)
        self.ema(period=200)
        self.gain_loss()
        self.sma(period=14, result_column_name='AvgGain', input_column_name='Gain')
        self.sma(period=14, result_column_name='AvgLoss', input_column_name='Loss')
        self.rel_strength()
        self.rsi('RSI')
        self.rsi_max(period=14, result_column_name='RSI_Highest_High', input_column_name='RSI')
        self.rsi_min(period=14, result_column_name='RSI_Lowest_Low', input_column_name='RSI')
        self.stoch_rsi('14_Day_Stoch_RSI')
        self.sma(period=2, result_column_name='RSI_K_Line', input_column_name='14_Day_Stoch_RSI')
        self.sma(period=4, result_column_name='RSI_D_Line', input_column_name='14_Day_Stoch_RSI')
        self.ema_comparisions()
        stock_info_df = self.buy_condition()

        self.logfiler.info("Ending Calc Of Indicators")

        return stock_info_df

    def check_signals(self) -> Union[pd.DataFrame, None]:
        """Checks to see if any signals have been generated.

        Returns:
        ----
        {Union[pd.DataFrame, None]} -- If signals are generated then a pandas.DataFrame
            is returned otherwise nothing is returned.
        """

        signals_df = self._stock_frame._check_signals(
            indicators=self._indicator_signals,
            indciators_comp_key=self._indicators_comp_key,
            indicators_key=self._indicators_key
        )

        return signals_df
