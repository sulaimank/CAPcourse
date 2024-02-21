import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from scipy import stats

# Load the dataset
df = pd.read_csv('C:/Users/warup/OneDrive/Documents/School/UCF/B_Graduate/Spring24/CAP5619/W6/Mainproject/CAPcourse/datasets/carSales.csv')

# Data cleaning
## Handle missing values
df.fillna(method='ffill', inplace=True) # Example: forward fill
## Remove duplicate records
df.drop_duplicates(inplace=True)
## Convert data types
df['Annual Income'] = df['Annual Income'].astype(int)
df['Date'] = pd.to_datetime(df['Date'])
## Clean Car_id data to numerical
df['Car_id'] = df['Car_id'].str.replace('C_CND_', '')
# Select only numeric columns for the correlation matrix
#numeric_df = df.select_dtypes(include=[np.number])
## Convert 'Gender' to numeric: Male=0, Female=1
df['Gender_numeric'] = df['Gender'].map({'Male': 0, 'Female': 1})
## Convert 'Transmission' to numeric: Manual=0, Auto=1
df['Transmission_numeric'] = df['Transmission'].map({'Manual': 0, 'Auto': 1})
## select usable data for numeric correlation
columns_of_interest = ['Annual Income', 'Price ($)', 'Gender_numeric', 'Transmission_numeric']
subset_df = df[columns_of_interest]
## Normalize numerical features
scaler = StandardScaler()
df['Annual Income'] = scaler.fit_transform(df[['Annual Income']])
## Handle outliers (example using Z-score)
print(f"Dataset dimensions: {df.shape}")

# Provide descriptive statistics
print(df.describe(include='all'))

# Data Visualizations
## Box plot for Annual Income
sns.boxplot(x='Annual Income', y='Gender', native_scale=True, data=df)
plt.show()
## Scatter plot for Price vs. Annual Income
sns.scatterplot(x='Annual Income', y='Price ($)', hue='Gender', data=df)
plt.show()
## Heatmap for correlation
sns.heatmap(subset_df.corr(), annot=True, cmap='coolwarm')
plt.show()

# Data Transforms
## Encoding categorical variables
encoder = OneHotEncoder(sparse_output=False)
encoded_features = encoder.fit_transform(df[['Gender', 'Transmission']])
## PCA for dimensionality reduction
pca = PCA(n_components=2)
df_pca = pca.fit_transform(df.select_dtypes(include=[np.number]))