# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 10:00:10 2018

@author: Li
"""

import pandas_datareader as pdr
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from scipy import stats

# obtain data from yahoo
def get(tickers, startdate, enddate):
    def data(ticker):
        return (pdr.get_data_yahoo(ticker, start = startdate, end = enddate))
    datas = map (data, tickers)
    return(pd.concat(datas, keys = tickers, names = ['Ticker', 'Date']))
    
# obtain train and test datasets
tickers = ['^NYA', '^N225', '000001.SS', '^GSPC']
mkt_index = get(tickers, datetime.datetime(2010, 1, 1), datetime.datetime(2015, 9, 30))
mkt_index_val = get(tickers, datetime.datetime(2015, 10, 1), datetime.datetime(2017, 3, 31))
mkt_index_test = get(tickers, datetime.datetime(2017, 4, 1), datetime.datetime(2018, 10, 21))

# calculate daily pct change of returns
mkt_index_close = mkt_index[['Adj Close']]
mkt_index_returns = mkt_index_close.pct_change()
mkt_index_test_close = mkt_index_test[['Adj Close']]
mkt_index_test_returns = mkt_index_test_close.pct_change()
mkt_index_val_close = mkt_index_val[['Adj Close']]
mkt_index_val_returns = mkt_index_val_close.pct_change()

return_datas = mkt_index_returns.reset_index().pivot('Date', 'Ticker')
return_datas.columns = ['SHCOMP', 'SPX', 'NIKKEI', 'NYSE']
return_datas = return_datas.iloc[1:len(return_datas)-1]
return_datas.fillna(0, inplace = True)

nyse_close = return_datas[['NYSE']]
nikkei_close = return_datas[['NIKKEI']]

return_datas_test = mkt_index_test_returns.reset_index().pivot('Date', 'Ticker')
return_datas_test.columns = ['SHCOMP', 'SPX', 'NIKKEI', 'NYSE']
return_datas_test = return_datas_test.iloc[1:]
return_datas_test.fillna(0, inplace = True)

return_datas_val = mkt_index_val_returns.reset_index().pivot('Date', 'Ticker')
return_datas_val.columns = ['SHCOMP', 'SPX', 'NIKKEI', 'NYSE']
return_datas_val = return_datas_val.iloc[1:]
return_datas_val.fillna(0, inplace = True)
return_datas_val.head()

return_datas['NYSE'].hist(bins = 50, figsize = (3, 2))
#nikkei_returns.hist(bins = 50, figsize = (3, 2))
#shcomp_returns.hist(bins = 50, figsize = (3, 2))
plt.show()

# OLS regression
x = sm.add_constant(return_datas['NYSE'])
linear_model = sm.OLS(return_datas['NIKKEI'], x).fit()
linear_model.summary()

nyse_lag = nyse_close.shift(1)
nyse_lag.fillna('0', inplace = True)

from sklearn.model_selection import train_test_split
nyse_train, nyse_test, nikkei_train, nikkei_test = train_test_split(nyse_lag, nikkei_close, test_size = 0.25, random_state = 42)
from sklearn.preprocessing import PolynomialFeatures as pf
from sklearn.linear_model import LinearRegression
poly = pf(degree = 1, include_bias = False)
nyse_new = poly.fit_transform(nyse_train)
nyse_test_new = poly.fit_transform(nyse_test)
model = LinearRegression()
model.fit(nyse_new, nikkei_train)
nikkei_pred = model.predict(nyse_new)
nikkei_test_pred = model.predict(nyse_test_new)

plt.scatter(nyse_train, nikkei_train)
plt.plot(nyse_new[:, 0], nikkei_pred, 'r')
plt.plot(nyse_test_new[:, 0], nikkei_test_pred, 'g')
plt.legend(['Predicted line', 'Test data', 'Observed data'])
plt.show()

return_datas_test['nyse_test_lag'] = return_datas_test['NYSE'].shift(1)
return_datas_test.fillna(0, inplace = True)
nyse_test_lag = pd.DataFrame(return_datas_test['nyse_test_lag'])
poly_nyse_test_lag = poly.fit_transform(nyse_test_lag)

poly = pf(degree = 1, include_bias = False)
poly_nyse_test_lag = poly.fit_transform(nyse_test_lag)
model_lag = LinearRegression()
model_lag.fit(poly_nyse_lag, nikkei_close)
nikkei_pred_lag = model_lag.predict(poly_nyse_lag)

plt.scatter(return_datas_test['NYSE'], return_datas_test['NIKKEI'])
plt.plot(nyse_new[:, 0], nikkei_pred, 'r')
plt.plot(nyse_test_new[:, 0], nikkei_test_pred, 'g')
plt.legend(['Predicted line', 'Test data', 'Observed data'])
plt.show()

from xgboost import XGBClassifier
# XGBoost is an implementation of gradient boosted decision trees
xgmodel = XGBClassifier(max_depth=6, learning_rate=0.1,n_estimators=100,
                      n_jobs=16,scale_pos_weight=4,missing=np.nan,gamma=16,
                      eval_metric="auc",reg_lambda=40,reg_alpha=40)
xgmodel.fit(nikkei_train,nyse_train)


























