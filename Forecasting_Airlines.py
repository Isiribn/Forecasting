#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
data=pd.read_excel('Airlines+Data.xlsx')
data.head()


# In[3]:


data.tail()


# In[4]:


data.shape


# In[5]:


data.isnull().any()


# In[6]:


data.duplicated().sum()


# In[7]:


data.info()


# In[8]:


data.describe()


# In[9]:


data['Passengers'].value_counts().plot(kind='pie')


# In[10]:


data.plot()


# In[11]:


data.plot(kind='hist')


# In[12]:


from datetime import datetime
s=pd.Series(data['Month'])
data['Month']=pd.to_datetime(s, infer_datetime_format=True)
indexedData=data.set_index(['Month'])


# In[13]:


indexedData['1995-03']
indexedData['1995-03':'1995-06']
indexedData['1995']


# In[14]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.xlabel('Date')
plt.ylabel('Passengers')
plt.plot(indexedData)


# In[ ]:


pip install fbprophet


# In[17]:


train_dataset=pd.DataFrame()
train_dataset['ds'] = data['Month']
train_dataset['y']= data['Passengers']
train_dataset.head()


# In[18]:


train_dataset.shape


# In[19]:


from fbprophet import Prophet
prophet_basic = Prophet()
prophet_basic.fit(train_dataset)


# In[20]:


#Used for testing or predicting
future= prophet_basic.make_future_dataframe(periods=14, freq='M')
future.tail(15)


# In[21]:


future.shape


# In[22]:


#plot of forecast
forecast=prophet_basic.predict(future)
fig1 =prophet_basic.plot(forecast)


# In[23]:


fig2 = prophet_basic.plot_components(forecast)


# In[27]:


#Incaseof any changepoints like natural or man-made faults
from fbprophet.plot import add_changepoints_to_plot
pro_change= Prophet(n_changepoints=4)
forecast = pro_change.fit(train_dataset).predict(future)
fig3= pro_change.plot(forecast);
a = add_changepoints_to_plot(fig3.gca(), pro_change, forecast)


# In[ ]:





# In[72]:


m = Prophet(weekly_seasonality=False, daily_seasonality=False, n_changepoints=4)
m.add_seasonality(name='yearly', period=12, fourier_order=5)
m.fit(train_dataset)


# In[73]:


future = m.make_future_dataframe(periods=12, freq='M')
forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]


# In[74]:


figure = m.plot(forecast)


# In[75]:


fig_decompose = m.plot_components(forecast)


# # Reduce Fourier Order

# In[80]:


m2 = Prophet(weekly_seasonality=False, daily_seasonality=False, n_changepoints=4)
m2.add_seasonality(name='yearly', period=12, fourier_order=1)

m2.fit(train_dataset)
future2 = m2.make_future_dataframe(periods=12, freq='m')
forecast2 = m2.predict(future2)
forecast2[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()


# In[81]:


fig2 = m2.plot(forecast2)


# In[82]:


fig2_decompose = m2.plot_components(forecast2)


# # Performance Metrics

# In[83]:


forecast['cutoff'] = pd.to_datetime('1995-05-01')
forecast['y'] = train_dataset['y']
forecast.tail()


# In[84]:


df_p = performance_metrics(forecast)
df_p.head()


# In[85]:


df_p.shape


# In[86]:


plt.plot(df_p['rmse'])


# In[ ]:




