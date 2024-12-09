#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
ppt = pd.read_csv(r"C:\datasets\Property_Price_Train.csv")


# In[3]:


ppt.isnull().sum()[ppt.isnull().sum() > 0]


# In[4]:


ppt = ppt.drop(['Fireplace_Quality',  'Pool_Quality', 'Fence_Quality', 'Miscellaneous_Feature' , 'Lane_Type', 'Id'], axis=1)


# In[62]:


ppt.columns


# In[5]:


ppt.shape


# In[6]:


ppt.Lot_Extent=ppt.Lot_Extent.fillna(ppt.Lot_Extent.mean())

ppt.Brick_Veneer_Type=ppt.Brick_Veneer_Type.fillna('None')
ppt.Brick_Veneer_Area=ppt.Brick_Veneer_Area.fillna(ppt.Brick_Veneer_Area.mean())
ppt.Basement_Height=ppt.Basement_Height.fillna('TA')
ppt.Basement_Condition=ppt.Basement_Condition.fillna('TA')
ppt.Exposure_Level=ppt.Exposure_Level.fillna('No')
ppt.BsmtFinType1=ppt.BsmtFinType1.fillna('Unf')
ppt.BsmtFinType2=ppt.BsmtFinType2.fillna('Unf')
ppt.Electrical_System=ppt.Electrical_System.fillna('SBrkr')

ppt.Garage=ppt.Garage.fillna('Attchd')
ppt.Garage_Built_Year=ppt.Garage_Built_Year.fillna(ppt.Garage_Built_Year.mean())
ppt.Garage_Finish_Year=ppt.Garage_Finish_Year.fillna('Unf')
ppt.Garage_Quality=ppt.Garage_Quality.fillna('TA')
ppt.Garage_Condition=ppt.Garage_Condition.fillna('TA')


# In[ ]:





# In[7]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


# In[8]:


ppt[ppt.select_dtypes(include= 'object').columns]= ppt[ppt.select_dtypes(include= 'object').columns].apply(le.fit_transform)


# In[9]:


ppt.info()


# In[10]:


ppt.head()


# In[42]:


ppt = df1 


# In[43]:


from sklearn.model_selection import train_test_split
train_ppt, test_ppt = train_test_split(ppt , test_size= 0.2)


# In[44]:


train_ppt_x = train_ppt.drop(['Sale_Price'], axis=1)
train_ppt_y = train_ppt.Sale_Price
test_ppt_x = test_ppt.drop(['Sale_Price'], axis=1)
test_ppt_y = test_ppt.Sale_Price


# In[45]:


from sklearn.linear_model import LinearRegression
linreg = LinearRegression()


# In[46]:


linreg.fit(train_ppt_x, train_ppt_y)


# In[47]:


Rsq =linreg.score(train_ppt_x, train_ppt_y)
Rsq


# In[48]:


N = train_ppt_x.shape[0]
K = train_ppt_x.shape[1]
adj_Rsq =1 - (1 -Rsq )*(N-1)/(N-K-1) 
adj_Rsq


# In[49]:


pred_train_ppt = linreg.predict(train_ppt_x)
err_train_ppt = train_ppt_y - pred_train_ppt
err_train_ppt.mean()


# In[50]:


import numpy as np
np.round(2.987693992045157e-11)


# In[51]:


err_train_ppt.skew()


# In[52]:


err_train_ppt.kurtosis()


# In[53]:


import matplotlib.pyplot as plt

plt.plot(err_train_ppt, '.')



# In[54]:


plt.hist(err_train_ppt, bins= 50 , edgecolor = 'green');


# In[55]:


import seaborn as sns
sns.regplot(x=train_ppt_y, y=pred_train_ppt, data= train_ppt, color='green')


# In[56]:


mse_train_ppt = np.mean(np.square(err_train_ppt))
mse_train_ppt


# In[57]:


mape_train_ppt = np.mean(np.abs(err_train_ppt*100/train_ppt_y))
mape_train_ppt


# In[58]:


pred_test = linreg.predict(test_ppt_x)
err_test_ppt = test_ppt_y-pred_test


# In[59]:


mse_test_ppt = np.mean(np.sum(err_test_ppt))
mse_test_ppt


# In[60]:


rmse_test_ppt = np.sqrt(mse_test_ppt)
rmse_test_ppt


# In[61]:


mape_test_ppt = np.mean(np.abs(err_test_ppt*100/test_ppt_y))
mape_test_ppt


# In[ ]:





# In[ ]:





# In[39]:


def remove_outliers(df , col, k):
    mean = df[col].mean()
    global df1
    sd = df[col].std()
    final_list = [x for x in df[col] if(x > mean-k*sd)]
    final_list = [x for x in df[col] if(x < mean+k*sd)]
    df1 = df.loc[df[col].isin(final_list)]; 
    print(df1.shape)
    print("number of outliers removed:" , df.shape[0]-df1.shape[0])


# In[40]:


remove_outliers(ppt, 'Sale_Price', 2)


# In[41]:


df1.shape


# In[ ]:


# add ppt = df1 just before sampling and run the code inly after that


# In[ ]:





# In[ ]:




