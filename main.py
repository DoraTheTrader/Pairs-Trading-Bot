import numpy as np
import pandas as pd
import seaborn
import statsmodels
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint, adfuller

import matplotlib.pyplot as plt
import seaborn as sns;

sns.set(style="whitegrid")

pd.core.common.is_list_like = pd.api.types.is_list_like
from pandas_datareader import data as pdr
import datetime
import yfinance as yf

yf.pdr_override()


# finding cointegrated pairs
def find_cointegrated_pairs(data):
    n = data.shape[1]
    score_matrix = np.zeros((n, n))
    pvalue_matrix = np.ones((n, n))
    keys = data.keys()
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            S1 = data[keys[i]]
            S2 = data[keys[j]]
            result = coint(S1, S2)
            score = result[0]
            pvalue = result[1]
            score_matrix[i, j] = score
            pvalue_matrix[i, j] = pvalue
            if pvalue < 0.05:
                pairs.append((keys[i], keys[j]))
    return score_matrix, pvalue_matrix, pairs


# giving input
start = datetime.datetime(2013, 1, 1)
end = datetime.datetime(2022, 10, 10)

tickers = ['AAPL', 'ADBE', 'ORCL', 'EBAY', 'MSFT', 'QCOM', 'HPQ', 'JNPR', 'AMD', 'IBM', 'SPY']

df = pdr.get_data_yahoo(tickers, start, end)['Close']
df.tail()

# finding p-value
scores, pvalues, pairs = find_cointegrated_pairs(df)
# print("pairs: ", pairs)

S1 = df['AMD']
S2 = df['MSFT']

score, pvalue, _ = coint(S1, S2)
# print(pvalue)
# if the p-value is less than 0.05, it means that thoose pairs are cointegrated

# calculating the spread
S1 = sm.add_constant(S1)
results = sm.OLS(S2, S1).fit()
S1 = S1['AMD']
b = results.params['AMD']

spread = S2 - b * S1

# examining the ration between the two time series
ratio = S1 / S2


# calculating the z-score. z-score is the number of standard deviations a datapoint is from the mean.
def zscore(series):
    return (series - series.mean()) / np.std(series)


# trade signal setup
ratios = df['AMD'] / df['MSFT']
train = ratios[:2461]
test = ratios[2461:]

# setting up indicators
ratios_mavg5 = train.rolling(window=5, center=False).mean()
ratios_mavg60 = train.rolling(window=60, center=False).mean()
std_60 = train.rolling(window=60, center=False).std()

# we can see from here that how much has the price moved away from the mean
zscore_60_5 = (ratios_mavg5 - ratios_mavg60) / std_60

plt.figure(figsize=(12, 6))
zscore_60_5.plot()
plt.xlim('2013-03-25', '2022-10-10')
plt.axhline(0, color='black')
plt.axhline(1.0, color='red', linestyle='--')
plt.axhline(-1.0, color='green', linestyle='--')
plt.legend(['Rolling Ratio z-Score', 'Mean', '+1', '-1'])
plt.show()

# plotting trade signals
plt.figure(figsize=(12, 6))

train[1722:].plot()
buy = train.copy()
sell = train.copy()
buy[zscore_60_5 > -1] = 0
sell[zscore_60_5 < 1] = 0
buy[1722:].plot(color='g', linestyle='None', marker='^')
sell[1722:].plot(color='r', linestyle='None', marker='^')
x1, x2, y1, y2 = plt.axis()
plt.axis((x1, x2, ratios.min(), ratios.max()))
plt.xlim('2019-01-01', '2022-10-10')
plt.legend(['Ratio', 'Buy Signal', 'Sell Signal'])
plt.show()

plt.figure(figsize=(12, 7))
S1 = df['ADBE'].iloc[:2461]
S2 = df['MSFT'].iloc[:2461]

S1[60:].plot(color='b')
S2[60:].plot(color='c')
buyR = 0 * S1.copy()
sellR = 0 * S1.copy()

# When you buy the ratio, you buy stock S1 and sell S2
buyR[buy != 0] = S1[buy != 0]
sellR[buy != 0] = S2[buy != 0]

# When you sell the ratio, you sell stock S1 and buy S2
buyR[sell != 0] = S2[sell != 0]
sellR[sell != 0] = S1[sell != 0]

buyR[60:].plot(color='g', linestyle='None', marker='^')
sellR[60:].plot(color='r', linestyle='None', marker='^')
x1, x2, y1, y2 = plt.axis()
plt.axis((x1, x2, min(S1.min(), S2.min()), max(S1.max(), S2.max())))
plt.ylim(25, 705)
plt.xlim('2013-03-22', '2022-10-10')

plt.legend(['AMD', 'MSFT', 'Buy Signal', 'Sell Signal'])
plt.show()


# Trade using a simple strategy
def trade(S1, S2, window1, window2):
    # If window length is 0, algorithm doesn't make sense, so exit
    if (window1 == 0) or (window2 == 0):
        return 0

    # Compute rolling mean and rolling standard deviation
    ratios = S1 / S2
    ma1 = ratios.rolling(window=window1,
                         center=False).mean()
    ma2 = ratios.rolling(window=window2,
                         center=False).mean()
    std = ratios.rolling(window=window2,
                         center=False).std()
    zscore = (ma1 - ma2) / std

    # Simulate trading
    # Start with no money and no positions
    money = 0
    countS1 = 0
    countS2 = 0
    for i in range(len(ratios)):
        # Sell short if the z-score is > 1
        if zscore[i] < -1:
            money += S1[i] - S2[i] * ratios[i]
            countS1 -= 1
            countS2 += ratios[i]
            # print('Selling Ratio %s %s %s %s'%(money, ratios[i], countS1,countS2))
        # Buy long if the z-score is < -1
        elif zscore[i] > 1:
            money -= S1[i] - S2[i] * ratios[i]
            countS1 += 1
            countS2 -= ratios[i]
            # print('Buying Ratio %s %s %s %s'%(money,ratios[i], countS1,countS2))
        # Clear positions if the z-score between -.5 and .5
        elif abs(zscore[i]) < 0.75:
            money += S1[i] * countS1 + S2[i] * countS2
            countS1 = 0
            countS2 = 0
            # print('Exit pos %s %s %s %s'%(money,ratios[i], countS1,countS2))

    return money


print(trade(df['AMD'].iloc[1722:], df['MSFT'].iloc[1722:], 60, 5))





