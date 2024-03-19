import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.naive_bayes import MultinomialNB

# Load the data
## Change file path to where you have dataset saved

df = pd.read_csv(r"C:\Users\Setup User\Downloads\carSales.csv")
#df = pd.read_csv('carSales.csv')

df = df.drop(columns=['Customer Name','Dealer_Name','Phone','Company','Dealer_No ','Car_id','Dealer_Region'])
df.head(5)
# Convert the Date column to datetime
df['Date'] = pd.to_datetime(df['Date'])

# check for na or Null values
isnacount = df.isna().sum()
isnullcount = df.isnull().sum()

# print results
print("DataFrame data types: \n", df.dtypes)
print("NA data counts: \n", isnacount)
print("\nNULL data counts: \n", isnullcount)
# Get the number of rows and columns
num_rows, num_columns = df.shape

# Print the number of rows and columns
print(f"Number of rows: {num_rows}")
print(f"Number of columns: {num_columns}")
# Setup column name variable and empty dictionary
columns = ['Body Style','Model', 'Engine', 'Color']
unique_values_dict = {}

for column in columns:
    unique_values_dict[column] = df[column].unique().tolist()

# Print the unique values for each column
for column, unique_values in unique_values_dict.items():
    print(f"{column}: {unique_values}")
# Change just the engine column to correct typo
df['Engine'] = df['Engine'].replace('DoubleÂ\xa0Overhead Camshaft', 'Double Overhead Camshaft')
# Setup column name variable and empty dictionary
columns = ['Body Style','Engine', 'Color','Transmission','Gender','Price ($)','Model']  # replace with your actual column names
unique_values_dict = {}

for column in columns:
    unique_values_dict[column] = df[column].unique().tolist()

# Print the unique values for each column
for column, unique_values in unique_values_dict.items():
    print(f"{column}: {unique_values}")
dfdesc = df.describe()
dfdesc.round(2)
# Visualize with basic dataframe without any preprocessing
plt.figure(figsize=(12, 6))
sns.violinplot(x='Body Style', y='Annual Income', data=df)
plt.title('Annual Income Distribution across Body Styles')
plt.xlabel('Body Style')
plt.ylabel('Annual Income')
plt.show()
# Calculate the Inter Quartile Range for Annual Income
Q1 = df['Annual Income'].quantile(0.25)
Q3 = df['Annual Income'].quantile(0.75)
IQR = Q3 - Q1

# Define the cutoff for outliers
cutoff = IQR * 1.5

# Determine the bounds for the outliers
lower_bound = Q1 - cutoff
upper_bound = Q3 + cutoff

df_outliers = df[(df['Annual Income'] < lower_bound) | (df['Annual Income'] > upper_bound)]

# Print the outliers
print("Annual Income class outliers:\n", df_outliers['Annual Income'].head(5))

# Filter the dataframe to remove outliers from 'Annual Income'
df = df[(df['Annual Income'] >= lower_bound) & (df['Annual Income'] <= upper_bound)]

# Check if any values in the filtered DataFrame fall outside the bounds
outliers_present = any((df['Annual Income'] < lower_bound) | (df['Annual Income'] > upper_bound))

# Print result and confirm process complete.
if outliers_present:
    print("Outliers were not removed correctly.")
else:
    print("Outliers were removed correctly.")
# Now visualized the data without outliers
plt.figure(figsize=(12, 6))
sns.violinplot(x='Body Style', y='Annual Income', data=df)
plt.title('Annual Income Distribution across Body Styles (Outliers Removed)')
plt.xlabel('Body Style')
plt.ylabel('Annual Income')
plt.show()
remappeddf = df.copy()
remappeddf.drop(columns=['Date'], inplace=True)

# Setup the label encoder
le = LabelEncoder()

# Fit and transform the 'Model' column
remappeddf['Model'] = le.fit_transform(remappeddf['Model'])

print(remappeddf.head(5))
le = LabelEncoder()

# Fit and transform the 'Model' column
remappeddf['Model'] = le.fit_transform(remappeddf['Model'])


categorical_columns = ['Body Style', 'Engine', 'Color', 'Transmission', 'Gender']  # Categorical columns
# Keeping only the categorical columns for encoding
remappeddf_categorical = remappeddf[categorical_columns]

encoder = OneHotEncoder()
df_encoded = encoder.fit_transform(remappeddf_categorical).toarray()
df_encoded = pd.DataFrame(df_encoded, columns=encoder.get_feature_names_out(categorical_columns))

# Combine the encoded DataFrame with the rest of the data and exlude the original categorical columns for redundancy sake
df_rest = remappeddf.drop(columns=categorical_columns)
df_combined = pd.concat([df_rest, df_encoded], axis=1)

# Print preview of the combined DataFrame
print(df_combined.head())
# Get the number of rows and columns
num_rows, num_columns = df_combined.shape

# Print the number of rows and columns
print(f"Number of rows: {num_rows}")
print(f"Number of columns: {num_columns}")

# Post processing statistics of the dataset/ dataframe
print("\nRemapped DataFrame: \n\n", df_combined.describe().round(3))  # This will give you count, mean, std, min, 25%, 50%, 75%, max
print('\nVariance values: \n\n', df_combined.var().astype('Float32'))  # Variance
print('\nSkew measurement: \n\n',df_combined.skew().round(4))  # Skewness
# Setup the scalar
scaler = MinMaxScaler()

# Select the columns to be normalized
columns_to_normalize = ['Annual Income','Price ($)']  # Replace with your actual column names

# Normalize the selected columns
df_normalized = df_combined.copy()
df_normalized[columns_to_normalize] = scaler.fit_transform(df_normalized[columns_to_normalize])

# Print the normalized dataframe
print(df_normalized.head())
print("Before sampling: \n", df_normalized.Gender_Male.value_counts())

# Separate majority and minority classes
df_majority = df_normalized[df_normalized.Gender_Male==1]
df_minority = df_normalized[df_normalized.Gender_Male==0]

# Downsample majority class
df_majority_downsampled = resample(df_majority,
                                 replace=False,    # sample without replacement
                                 n_samples=len(df_minority),     # to match minority class
                                 random_state=123) # reproducible results

# Combine minority class with downsampled majority class
df_downsampled = pd.concat([df_majority_downsampled, df_minority])



###
###
###


# Display new class counts
print("\nAfter sampling: \n",df_downsampled.Gender_Male.value_counts())
print(df_downsampled.head())

# Set up the X and y variables
X = df_downsampled.drop(columns=['Annual Income'])  # Replace with the actual columns for X
y = df_downsampled['Annual Income']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
y_test.dropna(inplace=True)
X_test.dropna(inplace=True)
y_train.dropna(inplace=True)
X_train.dropna(inplace=True)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1)

model = LinearRegression()
# model = MultinomialNB()

print(f"Length of X_test: {len(X_test)}")
print(f"Length of y_test: {len(y_test)}")

model.fit(X_train, y_train)
#X_test.dropna(inplace=True)
y_pred = model.predict(X_test)
print(f"Length of y_pred: {len(y_pred)}")

accuracy = accuracy_score(y_test, y_pred)
precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')


f1_score(y_test, y_pred, average='macro')
# f1_score(y_true, y_pred, average='micro')
# f1_score(y_true, y_pred, average='weighted')
# f1_score(y_true, y_pred, average=None)