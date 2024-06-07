
"""
Created on Fri Jun  7 21:50:05 2024

"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.datasets import load_iris
import numpy as np

# Load the Iris dataset
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target

# Map target labels to species names
species_map = {i: species for i, species in enumerate(iris.target_names)}
df['species'] = df['species'].map(species_map)

# Scatter Plot
plt.figure(figsize=(10, 6))
for species in df['species'].unique():
    subset = df[df['species'] == species]
    plt.scatter(subset['sepal length (cm)'], subset['sepal width (cm)'], label=species)
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.title('Scatter Plot of Sepal Length vs Sepal Width')
plt.legend()
plt.show()

# Histogram
plt.figure(figsize=(10, 6))
df['sepal length (cm)'].plot.hist(bins=20, alpha=0.5)
plt.xlabel('Sepal Length (cm)')
plt.title('Histogram of Sepal Length')
plt.show()

# Box Plot
plt.figure(figsize=(10, 6))
sns.boxplot(x='species', y='petal length (cm)', data=df)
plt.title('Box Plot of Petal Length by Species')
plt.show()

# Violin Plot
plt.figure(figsize=(10, 6))
sns.violinplot(x='species', y='petal width (cm)', data=df)
plt.title('Violin Plot of Petal Width by Species')
plt.show()

# Pair Plot
sns.pairplot(df, hue='species', markers=["o", "s", "D"])
plt.suptitle('Pair Plot of Iris Data', y=1.02)
plt.show()

# Heatmap
correlation_matrix = df.drop('species', axis=1).corr()
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Heatmap of Feature Correlations')
plt.show()
