# -*- coding: utf-8 -*-
"""
Created on Fri June  20 08:38:01 2024

@author: sumit dhivar
"""

#Zero Variance and near zero variance 
#If there is no variance in the feature, then ML model will not get any intelligence, so it is better to ignore those features 



#-----------------------------------------Missing Values---------------------------------------------------
import pandas as pd
import numpy as np
df = pd.read_csv('modified ethnic.csv') 
df.var


#Check for NULL values 
df.isna().sum()

from sklearn.impute import SimpleImputer
mean_imputer = SimpleImputer(missing_values=np.nan,strategy='mean')

df['Salaries'] = pd.DataFrame((mean_imputer.fit_transform(df[['Salaries']])))

df['Salaries']

df['Salaries'].isna().sum()#Here we are checking that the sum of null values=0


#------------------------------------------------------------------------------
import pandas as pd 
data = pd.read_csv('ethnic.csv')

data.head(10)
 
data.info()

#It gives size, null values, rows, columns and column dat.

data.describe()

data['Salaries_new'] = pd.cut(data['Salaries'],bins=[min(data["Salaries"]),
                data.Salaries.mean(),max(data.Salaries)],labels=["Low","High"])

data['Salaries_new']=pd.cut(data['Salaries'], bins=[min(data.Salaries),data.Salaries.quantile(0.25),data.Salaries.mean(), data.Salaries.quantile(0.75),max(data.Salaries)], labels=['group1','group2','group3','group4'])
data.Salaries_new.value_counts()


#============================================================================== 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
df=pd.read_csv('animal_category.csv')
df

#Check thhe shape 
df.shape

df.drop(['Index'],axis=1,inplace=True)

#Checking wheter the Index column is deleted or not
df

df_new = pd.get_dummies(df)

df_new
df_new.shape

df_new.columns

#here we are getting 10 rows and 14 columns 
#we are getting two columns for homely and gender, one column 

df_new.drop(['Gender_Male','Homly_Yes'],axis=1,inplace=True)

df_new.shape

#How we are getting 30,12 
df_new.rename(columns={'Gender_female':'Gender', 'Homly_No':'Homly'},inplace=True)




#==============================================================================
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt  

df = pd.read_csv('ethnic diversity.csv')
df.shape
df.columns

df_new = pd.get_dummies(df)

df_new.shape


#one hot encoder
import pandas as pd 
#import numpy as np 
#import matplotlib.pyplot as plt 
from sklearn.preprocessing import OneHotEncoder 
enc = OneHotEncoder()

#We will use ethinc diversity dataset 
df = pd.read_csv('ethnic diversity.csv')
df.columns

#We have Salaries and age as numerical column , llet us make them 
#at position 0 and 1 so to make further data preprocessing easy

df=df[['Salaries', 'age','Employee_Name', 'Position', 'State', 'Sex','MaritalDesc', 'CitizenDesc', 'EmploymentStatus', 'Department','Race']]
#Check the dataframe in variable explorer 
#We want only nominal data and ordinal data for processsing 
#Hence skipped 0 th and 1st ccolumn and applied to on hot encoder 
enc_df = pd.DataFrame(enc.fit_transform(df.iloc[:,2:]).toarray())
enc_df
#=======================================================================================
#Label Encoder 
from sklearn.preprocessing import LabelEncoder 
#Creating instance of label encoder
labelencoder = LabelEncoder() 
#split your data into input and output variables 
X=df.iloc[:,0:9] 
y=df['Race']
df.columns 
#We have nominal data Sex, Matial Desc, CitizenDesc,
#we want to convert to label encoder 
X['Sex'] = labelencoder.fit_transform(X['Sex'])
X['MaritalDesc'] = labelencoder.fit_transform(X['MaritalDesc'])
X['CitizenDesc'] = labelencoder.fit_transform(X['CitizenDesc'])

#label encoder y
y = labelencoder.fit_transform(y)
#This is going to create ana array , hence convert 
#it back to dataframe 
y = pd.DataFrame(y)

y = pd.DataFrame(y)
df_new = pd.concat([X,y] , axis=1)

#if you will see variable explorer , y do not have column name
#hence rename the column

df_new = df_new.rename(columns={0:'Race'})


#=======================================================================================
import pandas as pd 
import numpy as np 
from sklearn.preprocessing import StandardScaler
d = pd.read_csv('mtcars.csv')
d.describe()

a = d.describe()

#Initialoze the dollar
scalar=StandardScaler()
df = scalar.fit_transform(d) 
dataset = pd.DataFrame(df)

res = dataset.describe()
#here if you will check res, in variable environment then
