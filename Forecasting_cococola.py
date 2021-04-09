#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
data=pd.read_excel('CocaCola_Sales_Rawdata.xlsx')
data.head()


# In[2]:


data.shape


# In[3]:


data.isnull().any()


# In[4]:


data.duplicated().any()


# In[5]:


data.info()


# In[6]:


data.describe()


# In[7]:


data.hist()


# In[10]:


import matplotlib.pyplot as plt
data.Sales.plot(label="org")
for i in range(2, 10, 2):
    data["Sales"].rolling(i).mean().plot(label=str(i))
    plt.legend(loc=3)


# In[11]:


data.plot(kind='kde')


# In[12]:


import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import SimpleExpSmoothing # SES
from statsmodels.tsa.holtwinters import Holt # Holts Exponential Smoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing  
import statsmodels.graphics.tsaplots as tsa_plots
import statsmodels.tsa.statespace as tm_models
#from datetime import datetime,time


# In[13]:


tsa_plots.plot_acf(data.Sales,lags=10)
tsa_plots.plot_pacf(data.Sales)


# In[14]:


train=data.head(48)
test=data.tail(12)


# In[15]:


import numpy as np
def MAPE(pred,org):
        temp=np.abs((pred-org))*100/org
        return np.mean(temp)


# In[16]:


#Simple Exponential Smoothing
ses_model=SimpleExpSmoothing(train["Sales"]).fit()
pred_ses=ses_model.predict(start=test.index[0],end=test.index[-1])
MAPE(pred_ses,test.Sales)


# In[17]:


#Holt Exponential smoothing
hw_model=Holt(train["Sales"]).fit()
pred_hw=hw_model.predict(start=test.index[0], end=test.index[-1])
MAPE(pred_hw,test.Sales)


# In[18]:


hwe_model_add_add = ExponentialSmoothing(train["Sales"],seasonal="add",trend="add",seasonal_periods=4,damped=True).fit()
pred_hwe_add_add = hwe_model_add_add.predict(start = test.index[0],end = test.index[-1])
MAPE(pred_hwe_add_add,test.Sales)


# In[19]:


hwe_model_mul_add = ExponentialSmoothing(train["Sales"],seasonal="mul",trend="add",seasonal_periods=4).fit()
pred_hwe_mul_add = hwe_model_mul_add.predict(start = test.index[0],end = test.index[-1])
MAPE(pred_hwe_mul_add,test.Sales)


# In[20]:


plt.plot(train.index, train["Sales"], label='Train',color="r")


# In[21]:


plt.plot(test.index, test["Sales"], label='Test',color="blue")


# In[22]:


plt.plot(pred_ses.index, pred_ses, label='SimpleExponential',color="green")
plt.plot(pred_hw.index, pred_hw, label='Holts_winter',color="red")


# In[23]:


plt.plot(pred_hwe_add_add.index,pred_hwe_add_add,label="HoltsWinterExponential_1",color="brown")
plt.plot(pred_hwe_mul_add.index,pred_hwe_mul_add,label="HoltsWinterExponential_2",color="yellow")
plt.legend(loc='best')


# In[ ]:





# In[ ]:





# In[ ]:




