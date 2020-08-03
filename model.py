# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 06:10:29 2020

@author: gupta
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import seaborn as sns

df=pd.read_csv('employee-salary.csv')
df.info()

df=df.fillna(0.0)

#Converting blood groups to integer values
def blood_group_to_int(group):
    blood_dict={'O':0,'A':1,'AB':2,'B':3}    
    return blood_dict[group]
	
	
df['blood_group']=df['groups'].apply(lambda x : blood_group_to_int(x))
df.drop(columns=['id','groups'],inplace=True)
# Checking the field value counts again
df.info()

X=df[['age','healthy_eating','active_lifestyle','blood_group']]
y=df.salary

from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


#Model creation
from sklearn.linear_model import LinearRegression
linear = LinearRegression()
linear.fit(X_train,y_train)
pred=linear.predict(X_test)

# plotting regression line
sns.regplot(y_test,pred)

#Creating pickel file
pickle.dump(linear, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))


# field values- 'age','healthy_eating','active_lifestyle','blood_group'
model.predict([[40,7,4,0]])



