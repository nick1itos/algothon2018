
# coding: utf-8

# In[1]:


import pandas_datareader as pdr
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from scipy import stats


# In[2]:


# obtain data from yahoo
def get(tickers, startdate, enddate):
    def data(ticker):
        return (pdr.get_data_yahoo(ticker, start = startdate, end = enddate))
    datas = map (data, tickers)
    return(pd.concat(datas, keys = tickers, names = ['Ticker', 'Date']))


# In[42]:


# obtain train and test datasets
tickers = ['^NYA', '^N225', '000001.SS', '^GSPC']
mkt_index = get(tickers, datetime.datetime(2010, 1, 1), datetime.datetime(2015, 9, 30))
mkt_index_val = get(tickers, datetime.datetime(2015, 10, 1), datetime.datetime(2017, 3, 31))
mkt_index_test = get(tickers, datetime.datetime(2017, 4, 1), datetime.datetime(2018, 10, 21))


# In[44]:


mkt_index.head(1)


# In[4]:


# calculate daily pct change of returns
mkt_index_close = mkt_index[['Adj Close']]
mkt_index_returns = mkt_index_close.pct_change()
mkt_index_test_close = mkt_index_test[['Adj Close']]
mkt_index_test_returns = mkt_index_test_close.pct_change()
mkt_index_val_close = mkt_index_val[['Adj Close']]
mkt_index_val_returns = mkt_index_val_close.pct_change()


# In[5]:


return_datas = mkt_index_returns.reset_index().pivot('Date', 'Ticker')
return_datas.columns = ['SHCOMP', 'SPX', 'NIKKEI', 'NYSE']
return_datas = return_datas.iloc[1:len(return_datas)-1]
return_datas.fillna(0, inplace = True)
return_datas.tail()


# In[6]:


nyse_close = return_datas[['NYSE']]
nikkei_close = return_datas[['NIKKEI']]


# In[7]:


return_datas_test = mkt_index_test_returns.reset_index().pivot('Date', 'Ticker')
return_datas_test.columns = ['SHCOMP', 'SPX', 'NIKKEI', 'NYSE']
return_datas_test = return_datas_test.iloc[1:]
return_datas_test.fillna(0, inplace = True)
return_datas_test.head()


# In[8]:


return_datas_val = mkt_index_val_returns.reset_index().pivot('Date', 'Ticker')
return_datas_val.columns = ['SHCOMP', 'SPX', 'NIKKEI', 'NYSE']
return_datas_val = return_datas_val.iloc[1:]
return_datas_val.fillna(0, inplace = True)
return_datas_val.head()


# In[9]:


"""
#seperate indices returns
nyse_returns = mkt_index_returns.iloc[mkt_index_returns.index.get_level_values('Ticker') == '^NYA']
nyse_returns.index = nyse_returns.index.droplevel('Ticker')
#drop the first NA row
nyse_returns = nyse_returns.iloc[1:]

nikkei_returns = mkt_index_returns.iloc[mkt_index_returns.index.get_level_values('Ticker') == '^N225']
nikkei_returns.index = nikkei_returns.index.droplevel('Ticker')
nikkei_returns = nikkei_returns.iloc[1:]
shcomp_returns = mkt_index_returns.iloc[mkt_index_returns.index.get_level_values('Ticker') == '000001.SS']
shcomp_returns.index = shcomp_returns.index.droplevel('Ticker')
shcomp_returns = shcomp_returns.iloc[1:]
spx_returns = mkt_index_returns.iloc[mkt_index_returns.index.get_level_values('Ticker') == '^GSPC']
spx_returns.index = spx_returns.index.droplevel('Ticker')
spx_returns = spx_returns.iloc[1:]
"""


# In[10]:


"""
#concatenate indices returns
return_data = pd.concat([nyse_returns, nikkei_returns, shcomp_returns, spx_returns], axis = 1)
return_data.columns = ['NYSE', 'NIKKEI', 'SHCOMP', 'SPX']

#fill NA with 0
return_data.fillna(0, inplace = True)
"""


# In[11]:


return_datas['NYSE'].hist(bins = 50, figsize = (3, 2))
#nikkei_returns.hist(bins = 50, figsize = (3, 2))
#shcomp_returns.hist(bins = 50, figsize = (3, 2))
plt.show()


# In[12]:


return_datas.plot(figsize = (4, 3))
plt.show()


# In[14]:


# OLS regression
x = sm.add_constant(return_datas['NYSE'])
linear_model = sm.OLS(return_datas['NIKKEI'], x).fit()
linear_model.summary()


# In[46]:


pd.return_datas.corr('NYSE', 'NIKKEI')


# In[17]:


nyse_lag = nyse_close.shift(1)
nyse_lag.fillna('0', inplace = True)
nyse_lag.tail(2)
nikkei_close.tail(2)


# In[18]:


from sklearn.model_selection import train_test_split
nyse_train, nyse_test, nikkei_train, nikkei_test = train_test_split(nyse_lag, nikkei_close, test_size = 0.25, random_state = 42)


# In[20]:


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


# In[41]:


return_datas_test['nyse_test_lag'] = return_datas_test['NYSE'].shift(1)
return_datas_test.fillna(0, inplace = True)
nyse_test_lag = pd.DataFrame(return_datas_test['nyse_test_lag'])


# In[ ]:


poly = pf(degree = 1, include_bias = False)
poly_nyse_test_lag = poly.fit_transform(nyse_test_lag)
model_lag = LinearRegression()
model_lag.fit(nyse_test_lag, nikkei_close)
nikkei_pred_lag = model_lag.predict(poly_nyse_test_lag)


# In[29]:


plt.scatter(return_datas_test['NYSE'], return_datas_test['NIKKEI'])
plt.plot(nyse_new[:, 0], nikkei_pred, 'r')
plt.plot(nyse_test_new[:, 0], nikkei_test_pred, 'g')
plt.legend(['Predicted line', 'Test data', 'Observed data'])
plt.show()


# In[ ]:


from xgboost import XGBClassifier
# XGBoost is an implementation of gradient boosted decision trees
xgmodel = XGBClassifier(max_depth=6, learning_rate=0.1,n_estimators=100,
                      n_jobs=16,scale_pos_weight=4,missing=np.nan,gamma=16,
                      eval_metric="auc",reg_lambda=40,reg_alpha=40)
xgmodel.fit(nikkei_train,nyse_train)


# In[ ]:


from sklearn.metric import roc_auc_score
y_train_predcted = xgmodel.predict_proba()

