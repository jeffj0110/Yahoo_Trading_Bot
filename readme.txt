Nov 5, 2024 
This code was updated to run with Python version 3.12.7

# Setup to run this code
- You don't need any logins for Yahoo or Yahoo Finance
- Access to their market data is free, but it will be delayed
- please review the libraries required in 'requirements.txt' to 
      understand the python libraries which will need to be 
      installed (ie. pip install 'libraries')

# The code will take as input a ticker symbol (ie. AAPL), the start and end dates of the
# market data and the type of data to process with the algorithmic strategy.  
# The command line parameter examples -

cd <the directory where python code resides>
python Yahoo_GetOHLC.py -t MU -s 2019-01-21 -e 2022-01-21 -b 1d

python Yahoo_GetOHLC -t <tickersymbol> -p <period_string> -b <bar_string>
     <start_date> : a date in the format of YYYY-MM-DD that historical data should start from
     <end_date> : a date in the format of YYYY-MM-DD that historical data should end
     <bar string> : possible value-- 1min, 2min, 3min, 5min, 10min, 15min, 30min, 1h, 2h, 3h, 4h, 8h, 1d, 1w, 1m

This will generate two files in the directory the program is started from -
    <tickersymbol>_logfile_YYYY_MM_DD-HHMMSS.txt     : This is a timestamped log file
    <tickersymbol>_run_YYYY_MM_DD_HHMMSS.csv         : A comma delimited file market data and results
