#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics


# In[2]:


HouseDF = pd.read_csv (r'D:\py\test\machine_learning\Datasets\USA_Housing.csv')


# In[3]:


HouseDF.head() 


# In[32]:


HouseDF.describe()


# In[4]:


HouseDF.info()


# In[35]:


for col in HouseDF.columns:
    miss_data = HouseDF[col].isna().sum()
    miss_per =  miss_data/len(HouseDF)*100
    print(f"Columns: {col} has {miss_per}% missing data")


# In[45]:


fig,ax= plt.subplots(figsize=(8,5))
sns.heatmap(HouseDF.isna(),cmap="Blues", cbar =False, yticklabels =False)


# In[42]:


HouseDF.columns


# In[43]:


sns.pairplot(HouseDF)


# In[8]:


sns.distplot(HouseDF['Price'])


# In[9]:


sns.distplot(HouseDF['Avg. Area Income'])


# In[10]:


sns.distplot(HouseDF['Avg. Area House Age'])
sns.distplot(HouseDF['Avg. Area Number of Rooms'])


# In[11]:


sns.distplot(HouseDF['Avg. Area Number of Rooms'])


# In[12]:


sns.distplot(HouseDF['Avg. Area House Age'])


# In[13]:


sns.distplot(HouseDF['Avg. Area Number of Bedrooms'])


# In[14]:


sns.distplot(HouseDF['Area Population'])


# In[15]:


sns.heatmap(HouseDF.corr(), annot=True)


# In[16]:


X = HouseDF[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
               'Avg. Area Number of Bedrooms', 'Area Population']]

y = HouseDF['Price']


# In[17]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101) 


# In[18]:


lm = LinearRegression() 

lm.fit(X_train,y_train) 


# In[19]:


print(lm.intercept_)


# In[20]:


coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient']) 
coeff_df


# In[21]:


predictions = lm.predict(X_test)  
plt.scatter(y_test,predictions)


# In[22]:


sns.distplot((y_test-predictions),bins=50); 


# In[36]:


print('MAE:', metrics.mean_absolute_error(y_test, predictions)) 
print('MSE:', metrics.mean_squared_error(y_test, predictions)) 
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions))) 

