import pandas as pd
# Sample data creation
data = {
    'age': [25, 30, None, 22, 35],
    'income': [50000, 60000, 55000, None, 70000],
    'gender': ['male', 'female', 'female', 'male', None],
    'occupation': ['engineer', 'doctor', 'artist', 'engineer', 'doctor']
}
# Create a DataFrame
df = pd.DataFrame(data)
# Impute numerical columns with median
from sklearn.impute import SimpleImputer
num_imputer = SimpleImputer(strategy='median')
df[['age', 'income']] = num_imputer.fit_transform(df[['age', 'income']])
# Impute categorical columns with most frequent
cat_imputer = SimpleImputer(strategy='most_frequent')
df[['gender', 'occupation']] = cat_imputer.fit_transform(df[['gender', 'occupation']])
# Save the modified DataFrame to a new CSV file
df.to_csv('imputed_dataset.csv', index=False)
print("CSV file 'imputed_dataset.csv' has been created successfully.")
