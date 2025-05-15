from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2


df=pd.read_csv('imputed_dataset.csv')
print(df.head())
num_imputer = SimpleImputer(strategy='median')
df[['age', 'income']] = num_imputer.fit_transform(df[['age', 'income']])

# Impute categorical columns with most frequent
cat_imputer = SimpleImputer(strategy='most_frequent')
df[['gender', 'occupation']] = cat_imputer.fit_transform(df[['gender', 'occupation']])

# One-hot encode
encoder = OneHotEncoder(sparse_output=False, drop='first')  # drop='first' to avoid dummy trap
encoded = encoder.fit_transform(df[['gender', 'occupation']])

# Convert to DataFrame
encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(['gender', 'occupation']))
df = pd.concat([df.drop(['gender', 'occupation'], axis=1), encoded_df], axis=1)

print(df.head())

# Standardize numerical columns
scaler = StandardScaler()
df[['age', 'income']] = scaler.fit_transform(df[['age', 'income']])

print(df.head())
