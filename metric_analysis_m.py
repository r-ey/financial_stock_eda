'''
EDA about what makes bank stock price soar high in the past 5 years
Data from https://www.nasdaq.com/market-activity/stocks/screener (Sector: Financial) -> bank_tickers.csv
This would be more like an oop approach
Remarks: NASDAQ Finance stock only (the recent 5y only)
Runtime: approx. 13-14 minutes
'''

import yfinance as yf
import pandas as pd
import numpy as np
import timeit
import collections
import math
from datetime import datetime
from scipy.signal import find_peaks
from scipy.interpolate import CubicSpline
from scipy import stats


start = timeit.default_timer()  # timer start

'''
Extracting data from bank_tickers.csv ()
Cleaning data to only choose from major bank industry
'''

df = pd.read_csv('data/bank_tickers.csv',
                header=0,
                usecols=['Symbol', 'Market Cap', 'Industry'])

df.columns = ['ticker', 'mc', 'industry']  # only need the symbol, market cap, and industry

df = df.dropna()
df = df.drop(df[(df.mc < 1) | (df.industry != 'Major Banks')].index)  # select from major banks only

'''
Labeling the tickers based on market cap [mega, large, medium, small, micro, nano]
- mega > $200B
- large $10B - $200B
- medium $2B - $10B
- small $300M - $2B
- micro $50M - $300M
- nano < $50M
'''

conditions = [
    df.mc >= 200_000_000_000,
    (df.mc >= 10_000_000_000) & (df.mc < 200_000_000_000),
    (df.mc >= 2_000_000_000) & (df.mc < 10_000_000_000),
    (df.mc >= 300_000_000) & (df.mc < 2_000_000_000),
    (df.mc >= 50_000_000) & (df.mc < 300_000_000),
    df.mc < 50_000_000
]

labels = ['Mega', 'Large', 'Medium', 'Small', 'Micro', 'Nano']

df['labels'] = np.select(conditions, labels, 'unknown')

'''
Removing all stocks that don't have complete information regarding balance sheet
Remarks: income statement and cashflow will have the same behaviour as balance sheet
'''

total_stocks = 0

for x in df['ticker']:
    data = yf.Ticker(x).get_balance_sheet()
    if data.empty:
        df = df.drop(df.index[df['ticker'] == x])
    else:
        total_stocks += 1


'''
Finding and plotting when the price has the most significant increase in the recent 5 years 
(due to yfinance)
Significant: searching peak with prominence of 15% of the ath
Remarks: find_peaks may fail sometimes especially if the overall graph isn't on upward trend
'''

tickers = []
prices = []
pairs = [[] for _ in range(total_stocks)]
hists = []

# saving tickers and prices for all banking stocks
# return: the corresponding ticker and price
def save_as_local_variable(sym):
    global tickers, prices, hists

    ticker = yf.Ticker(sym)
    check_hist = ticker.history(period='max').reset_index()

    if len(check_hist) < 1258:  # check if the company has been public for at least 5y
        hist = check_hist
    else:
        hist = ticker.history(period='5y').reset_index()
    
    price = hist['Close'].values

    tickers.append(ticker)
    prices.append(price)
    hists.append(hist)

    return ticker, price

# finding all peak and valley from all stocks
# return: the peak and valley
def peak_and_valley_finder(sym):
    _, price = save_as_local_variable(sym)

    if len(price) == 0:  # check if the array is empty or not
        return [], []
    else:
        neg_price = price * -1

        prominence_price = max(price) * 0.15
        peak, _ = find_peaks(price, prominence=prominence_price)
        valley, _ = find_peaks(neg_price, prominence=prominence_price)

        return peak, valley

# finding the peak and valley pairs
def find_pairs():
    global pairs
    for i in range(total_stocks):
        p, v = peak_and_valley_finder(df['ticker'].values[i])

        if (len(p) > 1) & (len(v) > 1):  # pair the peak and valley
            n = m = 0  # n is for valley and m is for peak
            while (n < len(v)) & (m < len(p)):
                if v[n] < p[m]:
                    pairs[i].append([v[n], p[m]])
                    n += 1
                    m += 1
                else:
                    m += 1

find_pairs()

'''
Making a metric dictionary for each stocks
Remarks: trying to use quarterly finance info, but it seems something inside of yfinance broke down
'''

metric_dict = [{} for _ in range(total_stocks)]

# making a metric dictionary from each stock from balance sheet, income statement, and cash flow
def make_metric_dict():
    global metric_dict, tickers

    for i in range(total_stocks):  # Transforming each data frame to dictionary
        bs = tickers[i].get_balance_sheet().reset_index().set_index('index').T.to_dict() 
        cf = tickers[i].get_cashflow().reset_index().set_index('index').T.to_dict()
        inc = tickers[i].get_income_stmt().reset_index().set_index('index').T.to_dict()
    
        # Merging and sorting the dictionary based on alphabetical order (keys)
        metric_dict[i] = collections.OrderedDict(sorted({**bs, **cf, **inc}.items()))

make_metric_dict()

'''
Fill the data gaps in the metric using cubic spline interpolation with natural boundary
since the metric only gives us at most 5 data points
'''

interpolates = [[] for _ in range(total_stocks)]

# finding the interpolation data
def cubic_interpolate():
    global metric_dict

    for i in range(total_stocks):
        for key in metric_dict[i]:
            current_year = int(datetime.now().year)
            values = list(metric_dict[i][key].values())
            # reverse since the data is backward, i.e, 2023 - 2019, we want 2019 - 2023
            # replacing NaN with 0
            y = [0 if math.isnan(z) else z for z in values][::-1]
            x = [current_year-4, current_year-3, current_year-2, current_year-1, current_year]

            # homogenized length of x and y if different
            if len(x) != len(y):
                x = x[:len(y)]

            cs = CubicSpline(x, y, bc_type='natural')
            x_new = np.linspace(0, 1257, 1258)  # 5 years of historical price
            y_new = cs(x_new)

            interpolates[i].append(y_new)

cubic_interpolate()

'''
Finding the correlated between metric and stock price using Kendall's Tau
'''

correlated_metrics = [{} for _ in range(total_stocks)]

# finding the mean correlation coefficient for each metric
def find_kendall_tau():
    global correlated_metrics

    for i in range(total_stocks):
        metric_name = list(metric_dict[i].keys())

        # 2 because correlation coefficient can' be more than 1
        correlated_metrics[i] = dict.fromkeys(metric_name, float('nan'))  
        x = []
        y = []

        for j in range(len(interpolates[i])):
            
            tau = 0

            for pair in pairs[i]:
                x.append(prices[i][pair[0]:pair[1]])  # stock price
                y.append(interpolates[i][j][pair[0]:pair[1]])  # metric interpolation

            if len(x) != 0:

                for k in range(len(x)):  # find kendall's tau for each pair
                    res = stats.kendalltau(x[k], y[k])
                    tau += res.statistic  # tau coefficient

                correlated_metrics[i][metric_name[j]] = tau / len(x)

find_kendall_tau()

'''
Conclusion on its price and its metrics
Remarks: correlation doesn't imply causation, we just want to find a pattern using stock metric
'''

# print all the conclusion especially the kendall's tau coefficient
def conclusion():
    print('EDA CONCLUSION')
    for i in range(total_stocks):
        header = '--- ' + df['ticker'].values[i] + ' - ' + df['labels'].values[i] + ' Bank ---'
        print(header)
        for key in correlated_metrics[i]:
            print(key + ': ' + str(correlated_metrics[i][key]))
        print('------------------------------')

# conclusion()

stop = timeit.default_timer()  # timer end
print('Time: ', stop - start)
