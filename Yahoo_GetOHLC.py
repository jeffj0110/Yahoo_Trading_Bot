import sys, getopt, os
from datetime import datetime
from datetime import timedelta
from datetime import timezone
import pytz
import logging
import m_logger
import pandas as pd

from functions import setup_func
from indicator_calcs import Indicators

def Calc_Returns(stock_info, start_date, end_date, logger):

    column_strategy_return = []
    column_bnh_return = []
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    days_between = (end_dt - start_dt).days

    final_pnl_value = stock_info['Running_PnL'][-1]
    starting_closing_price = stock_info['close'][0]
    annualized_return_on_strategy = ((1+(final_pnl_value/starting_closing_price)) ** (365/days_between)) - 1
    end_closing_price = stock_info['close'][-1]
    annualized_buy_hold_return = ((1+((end_closing_price-starting_closing_price)/starting_closing_price)) ** (365/days_between)) - 1
    column_strategy_return.append(str(round((annualized_return_on_strategy * 100),2)))
    column_bnh_return.append(str(round((annualized_buy_hold_return * 100),2)))
    for i in range(1,len(stock_info)) :
        column_strategy_return.append('')
        column_bnh_return.append('')

    stock_info['Strategy_Return'] = pd.Series(column_strategy_return).values
    stock_info['Buy_Hold_Return'] = pd.Series(column_bnh_return).values

    logger.info('Annualized Strategy Return = {strat_ret}'.format(strat_ret = column_strategy_return[0]))
    logger.info('Annualized Buy and Hold Return = {bnh_ret}'.format(bnh_ret = column_bnh_return[0]))

    return stock_info


def Get_Data(inputsymbol, start_date, end_date, bar_string, logger, now) :

    symbol = inputsymbol

    # Sets up the robot class, robot's portfolio, and the TDSession object
    trading_robot = setup_func(logger)

    # Grab the historical prices for the symbol we're trading.

    historical_prices_df = trading_robot.grab_historical_prices(
        start=start_date,
        end=end_date,
        bar_string=bar_string,
        symbols=[symbol]
    )
    logger.info("Data Retrieval From Yahoo Finance Complete")

    # Get current date and create the excel sheet name
    # J. Jones - changed default timezone to EST
    # added seconds to the file name
    filename = "{}_run_{}".format(symbol, now)
    #json_path = './config'
    full_path = filename + ".csv"

    # Convert data to a Data StockFrame.
    if len(historical_prices_df) == 0 :
        logger.info("No Data Returned")
        sys.exit(-1)

    # Create an indicator Object.
    indicator_client = Indicators(input_price_data_frame=trading_robot.stock_frame, lgfile=logger)

    stock_info_df = indicator_client.refresh()

    stock_info_df = Calc_Returns(stock_info_df, start_date, end_date, logger)

    logger.info('Starting to write spreadsheet to {fname}'.format(fname=full_path))

    # Save an excel sheet with the data
    stock_info_df.to_csv(full_path)

    logger.info('Finished writing spreadsheet')

    return stock_info_df




# Check to see if this file is being executed as the "Main" python
# script instead of being used as a module by some other python script
# This allows us to use the module which ever way we want.
#
def main(argv):
    inputsymbol = ''
    bar_string = ''
    start_date = ''
    end_date = ''

#    Yahoo_GetOHLC -t ticker -s <start_date> -e <end_date> -b <bar_string>
#    <start_date> : a date in the format of YYYY-MM-DD that historical data should start from
#    <end_date> : a date in the format of YYYY-MM-DD that historical data should end
#    bar string -- possible value-- 1min, 2min, 3min, 5min, 10min, 15min, 30min, 1h, 2h, 3h, 4h, 8h, 1d, 1w, 1m
    try:
       opts, args = getopt.getopt(argv,"ht:s:e:b:")
    except getopt.GetoptError:
       print('Yahoo_GetOHLC -t ticker -s <start_date> -e <end_date> -b <bar_string>')
       print('<start_date> : a date in the format of YYYY-MM-DD that historical data should start from')
       print('<end_date> : a date in the format of YYYY-MM-DD that historical data should end')
       print('<bar string> : possible value-- 1min, 2min, 3min, 5min, 10min, 15min, 30min, 1h, 2h, 3h, 4h, 8h, 1d, 1w, 1m')
       print('See Yahoo Finance For Data Limits')
       sys.exit(2)
    for opt, arg in opts:
       if opt == '-h':
          print('Yahoo_GetOHLC -t ticker -s <start_date> -e <end_date> -b <bar_string>')
          print('<start_date> : a date in the format of YYYY-MM-DD that historical data should start from')
          print('<end_date> : a date in the format of YYYY-MM-DD that historical data should end')
          print('<bar string> : possible value-- 1min, 2min, 3min, 5min, 10min, 15min, 30min, 1h, 2h, 3h, 4h, 8h, 1d, 1w, 1m')
          print('See Yahoo Finance For Data Limits')
          sys.exit()
       elif opt in ("-t", "-T"):
           inputsymbol = arg
       elif opt in ("-s"):
           start_date = arg
       elif opt in ("-e"):
           end_date = arg
       elif opt in ("-b") :
           bar_string = arg

    if inputsymbol == '' :
        inputsymbol = "No_Sym_Defined"
    if bar_string == '' :
        bar_string = "No_Bar_Defined"
    if start_date == '' :
        start_date = 'No_Start_Date_defined'
    if end_date == '' :
        end_date = 'No_End_Date_defined'
    # J. Jones
    # Setting up a logging service for the bot to be able to retrieve
    # runtime messages from a log file
    est_tz = pytz.timezone('US/Eastern')
    now = datetime.now(est_tz).strftime("%Y_%m_%d-%H%M%S")
    logfilename = "{}_logfile_{}".format(inputsymbol, now)
    logfilename = logfilename + ".txt"
    logger = m_logger.getlogger(logfilename)

    if inputsymbol == "No_Sym_Defined" :
       logger.info("Invalid Command Line Arguments")
       logger.info("Yahoo_GetOHLC -s symbol -p <period_string> -b <bar_string>")
       logger.info('<start_date> : a date in the format of YYYY-MM-DD that historical data should start from')
       logger.info('<end_date> : a date in the format of YYYY-MM-DD that historical data should end')
       logger.info(
           '<bar string> : possible value-- 1min, 2min, 3min, 5min, 10min, 15min, 30min, 1h, 2h, 3h, 4h, 8h, 1d, 1w, 1m')
       logger.info('Maximum of 2-3 year daily prices')
       exit()
    else :
       logger.info('Running With Ticker Symbol : {sym}, Start Date : {sdate}, End Date : {edate}, bar : {bar}'.format(sym=inputsymbol, sdate=start_date, edate=end_date, bar=bar_string))

    Get_Data(inputsymbol, start_date, end_date, bar_string, logger, now)

    return True

if __name__ == "__main__":
   main(sys.argv[1:])
