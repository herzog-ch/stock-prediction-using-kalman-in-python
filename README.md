# Using a Kalman filter for predicting stock prices in python

This is a prototype implementation for predicting stock prices using a Kalman filter.
<br>
A generic Kalman filter using numpy matrix operations is implemented in src/kalman_filter.py. The predict and update function
can be used in different projects.
<br>
The stock prices were loaded from yahoo finance. The class YahooFinanceData
implemented in src/yahoo_financedata.py loads the .csv file holding the
stock prices (e.g. for the company Infineon) and provides a function
"next_measurement" to iterate through all rows.
<br>
For predicting the stock price of the next day, a simple model for the 
stock price behaviour is used. The state vector of the filter holds the
current price and the velocity. The velocity is
the change of the stock price per day. The filter is updated every day with
the newest stock price measurement.<br>
The main.py script will also provide some plots for analyzing the filter
output. Obviously the results cannot be taken serious for trading
with stocks. The stock prices are used as example data for working with
Kalman filters.