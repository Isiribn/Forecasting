#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
data=pd.read_csv('PlasticSales.csv')
data.head()


# In[2]:


data.tail()


# In[3]:


data.shape


# In[4]:


sale=pd.read_csv('PlasticSales.csv',header=0, index_col=0, parse_dates=True)
sale.head()


# In[5]:


sale.tail()


# In[6]:


sale.shape


# In[7]:


sale.isnull().any()


# In[8]:


sale.info()


# In[9]:


sale.describe()


# In[10]:


sale.plot()


# # Testing for stationary

# In[11]:


from statsmodels.tsa.stattools import adfuller
test_result=adfuller(sale['Sales'])


# In[12]:


#Ho:It is non stationary
#H1:It is stationary

def adfuller_test(Sales):
    result=adfuller(Sales)
    labels = ['ADF Test Statistics','p-value','#Lags Used','Number of Observations Used']
    for value,label in zip(result,labels):
        print(label+':'+str(value))
    if result[1] <=0.05:
        print("strong evidence against the null hypothesis(Ho),reject the null hypothesis")
    else:
        print("weak evidence against the null hypothesis,time series has a unit root,indicating it is non stationary ")  


# In[13]:


adfuller_test(sale['Sales'])


# In[14]:


#Differencing
sale['Sales First Difference'] = sale['Sales']-sale['Sales'].shift(1)
sale['Sales First Difference']


# In[15]:


sale['Sales'].shift(1)


# In[16]:


sale['Seasonal First Difference']=sale['Sales']-sale['Sales'].shift(12)
sale['Seasonal First Difference']


# In[17]:


sale.head()


# In[18]:


# Again test dicky fuller test
adfuller_test(sale['Seasonal First Difference'].dropna())


# In[19]:


sale['Seasonal First Difference'].plot()


# In[ ]:


#Auto and Partial autocorelation Model


# In[20]:


from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
import statsmodels.api as sm


# In[24]:


import matplotlib.pyplot as plt
fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(sale['Seasonal First Difference'].iloc[13:],lags=20,ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(sale['Seasonal First Difference'].iloc[13:],lags=20,ax=ax2)


# # ARIMA model

# In[25]:


#for non seasonable data
#p=1,d=1,q=0 or 1
from statsmodels.tsa.arima_model import ARIMA


# In[26]:


model=ARIMA(sale['Sales'],order=(1,1,1))
model_fit=model.fit()


# In[27]:


model_fit.summary()


# In[28]:


import statsmodels.api as sm
model=sm.tsa.statespace.SARIMAX(sale['Sales'],order=(1,1,0),seasonal_order=(1,1,1,6))
results=model.fit()


# In[29]:


sale['forecast']=results.predict(start=50,end=103,dynamic=True)
sale[['Sales','forecast']].plot(figsize=(12,8))


# In[31]:


from pandas.tseries.offsets import DateOffset
future_dates=[sale.index[-1]+DateOffset(months=x)for x in range(0,24)]


# In[32]:


future_datest_df=pd.DataFrame(index=future_dates[1:],columns=sale.columns)


# In[33]:


future_datest_df.tail()


# In[34]:


future_df=pd.concat([sale,future_datest_df])


# In[35]:


future_df['forecast']=results.predict(start=40, end=120,dynamic=True)
future_df[['Sales','forecast']].plot(figsize=(12,8))


# In[ ]:




