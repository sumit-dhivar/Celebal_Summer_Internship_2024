# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 22:56:21 2024

@author: sumit
"""

import pandas as pd
import numpy as np

path = "titanic.csv"
titanic = pd.read_csv(path)

titanic

titanic.columns
# Explore the data
print("First few rows of the dataset:")
print(titanic.head())

print("\nGeneral information about the dataset:")
print(titanic.info())

# Basic statistics
print("\nBasic statistics of numerical columns:")
print(titanic.describe())

import plotly.express as px

# Age distribution
age_hist = px.histogram(titanic, x='Age', nbins=20, title='Age Distribution')
age_hist.show()

# Fare distribution
fare_hist = px.histogram(titanic, x='Fare', nbins=50, title='Fare Distribution')
fare_hist.show()

# Pie chart for survival status
survival_pie = px.pie(titanic, names='Survived', title='Survival Status', labels={'survived': 'Survived'}, 
                      color_discrete_sequence=px.colors.qualitative.Set3)
survival_pie.show()

# Bar chart for survival count by passenger class
survival_bar_class = px.bar(titanic.groupby(['Pclass', 'Survived']).size().reset_index(name='count'), 
                            x='Pclass', y='count', color='Survived', barmode='group', 
                            title='Survival Count by Passenger Class', 
                            labels={'Pclass': 'Passenger Class', 'survived': 'Survived', 'count': 'Count'})
survival_bar_class.show()

#Data Preprocessing 
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler   

# Split data into training and test sets
train_data = titanic.dropna(subset=['Age'])  # Training data (non-missing values)
test_data = titanic[titanic['Age'].isnull()]  # Test data (missing values)

# Select features and target variable
X_train = train_data[['Pclass', 'Fare', 'SibSp', 'Parch']]  # Features
y_train = train_data['Age']  # Target variable
X_test = test_data[['Pclass', 'Fare', 'SibSp', 'Parch']]  # Features for test data







