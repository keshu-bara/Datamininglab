#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


salary = pd.read_csv('https://github.com/ybifoundation/Dataset/raw/main/Salary%20Data.csv')


# In[6]:


#definging target(y) and feature(x)
y = salary['Salary']
X = salary[['Experience Years']]


# In[19]:


#train test split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test= train_test_split(X,y,train_size = 0.7,random_state =42)


# In[22]:


#Check shape of train and test sample
X_train.shape , X_test.shape , y_train.shape , y_test.shape


# In[24]:


#select model
from sklearn.linear_model import LinearRegression
model = LinearRegression()


# In[25]:


model.fit(X_train,y_train)


# In[26]:


model.intercept_


# In[27]:


model.coef_


# In[30]:


#predict model
y_pred = model.predict(X_test)


# In[31]:


y_pred


# In[33]:


#model accuracy
from sklearn.metrics import mean_absolute_error,mean_absolute_percentage_error, mean_squared_error


# In[34]:


mean_absolute_error(y_test,y_pred)


# In[35]:


mean_absolute_percentage_error(y_test,y_pred)


# In[36]:


mean_squared_error(y_test,y_pred)


# In[40]:


y_pred_line = model.predict(X)
y_pred_test = model.predict(X_test)


# In[ ]:





# In[41]:


# Scatter plot of actual vs predicted
import matplotlib.pyplot as plt
import seaborn as sns

# Plot actual data
plt.figure(figsize=(8, 6))
plt.scatter(X, y, color='blue', label='Actual Data')                   # all real data
plt.plot(X, y_pred_line, color='red', label='Regression Line')         # regression curve
plt.scatter(X_test, y_pred_test, color='green', label='Predicted Points')  # predicted points on test data

plt.xlabel("X")
plt.ylabel("y")
plt.title("Linear Regression: Actual vs Predicted with Regression Line")
plt.legend()
plt.grid(True)
plt.show()


# In[ ]:




