#!/usr/bin/env python
# coding: utf-8

# # importing the necessary libraries and loading the data set file

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df=pd.read_csv(r"C:\Users\KARTIKI\Downloads\student_info.csv")
df


# In[3]:


df.head()


# In[4]:


df.tail()


# In[5]:


df.isnull().sum()


# In[6]:


df.describe()


# In[7]:


df.info()


# # Data Visualization

# In[19]:


sns.jointplot(x="study_hours", y="student_marks", data=df,
                  kind="reg", truncate=False,
                  )


# # Prepair the Data for Machine Learning Algorithm

# In[ ]:


#Cleaning the Data


# In[36]:


df.isnull().sum()


# In[31]:


avg=df["study_hours"].mean()


# In[32]:


avg


# In[33]:


df=df.fillna(avg)


# In[34]:


df.head()


# # Split the Dataset

# In[40]:


x=df.drop("student_marks",axis=1)
y=df.drop("study_hours",axis=1)
print("shape of x=",x.shape,"\n","shape of y=",y.shape)


# # importing Machine learning models and lib

# In[43]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=51)
print("x train ",x_train.shape)
print("y train ",y_train.shape)
print("x test ",x_test.shape)
print("x test ",y_test.shape)


# # select a model and train it

# In[45]:


#y=m*x+c
from sklearn.linear_model import LinearRegression
lr=LinearRegression()


# In[47]:


lr.fit(x_train,y_train)


# In[49]:


lr.coef_


# In[50]:


lr.intercept_


# In[51]:


m=3.93571802
c=50.44735504
y=m*4+c
y


# In[57]:


lr.predict([[4]])[0][0].round(2)


# In[58]:


y_pred=lr.predict(x_test)
y_pred


# In[63]:


pd.DataFrame(np.c_[x_test,y_test,y_pred],columns=["stud_hours","stud_original_marks","stud_predicted_marks"])


# In[67]:


lr.score(x_test,y_test).round(3)


# In[68]:


plt.scatter(x_train,y_train)
plt.show()


# In[71]:


plt.scatter(x_test,y_test,color="red")
plt.plot(x_train,lr.predict(x_train),color="blue")
plt.show()


# # Save ML Model

# In[72]:


import joblib
joblib.dump(lr,"student_marks_predictor_model.pkl")


# In[73]:


model=joblib.load("student_marks_predictor_model.pkl")


# In[74]:


model.predict([[5]])


# In[75]:


model.predict([[5]])[0][0]


# In[ ]:




