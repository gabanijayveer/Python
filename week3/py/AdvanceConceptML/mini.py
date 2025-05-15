import pandas as pd

# Load dataset
df = pd.read_csv("telco_churn.csv")

# Quick look at data
print(df.shape)
print(df.head())
print(df.info())
print(df.isnull().sum())

from sklearn.impute import SimpleImputer

# Separate numeric and categorical columns
num_cols = df.select_dtypes(include=['int64', 'float64']).columns
cat_cols = df.select_dtypes(include=['object']).columns

# Impute numeric columns with median
num_imputer = SimpleImputer(strategy='median')
df[num_cols] = num_imputer.fit_transform(df[num_cols])

# Impute categorical columns with most frequent
cat_imputer = SimpleImputer(strategy='most_frequent')
df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

from sklearn.preprocessing import OneHotEncoder

# Select categorical columns (excluding target)
X_cat = df[cat_cols.drop('Churn')]

# One-hot encoding
encoder = OneHotEncoder(sparse_output=False, drop='first')
X_encoded = encoder.fit_transform(X_cat)

# Convert to DataFrame
encoded_df = pd.DataFrame(X_encoded, columns=encoder.get_feature_names_out(cat_cols.drop('Churn')))

# Drop old cat columns and add encoded ones
df = df.drop(columns=X_cat.columns)
df = pd.concat([df, encoded_df], axis=1)

from sklearn.preprocessing import StandardScaler

# Scale numeric features
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

from sklearn.feature_selection import SelectKBest

X = df.drop(columns='Churn')
y = df['Churn'].map({'No': 0, 'Yes': 1})  # Encode target

# Select top 10 features
from sklearn.feature_selection import SelectKBest, f_classif

selector = SelectKBest(score_func=f_classif, k=10)
X_selected = selector.fit_transform(X, y)

selected_columns = X.columns[selector.get_support()]
print("Selected Features:", selected_columns)


from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Plot
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='coolwarm', alpha=0.5)
plt.title("PCA - Customer Churn")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()

X_clean = pd.DataFrame(X_selected, columns=selected_columns)
y_clean = y

X_clean.to_csv("X_clean.csv", index=False)
y_clean.to_csv("y_clean.csv", index=False)
df = pd.read_csv('telco_churn.csv')
print(df.head())
