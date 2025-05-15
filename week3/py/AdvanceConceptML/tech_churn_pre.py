import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, chi2, f_classif

# Load dataset
df = pd.read_csv('telco_churn.csv')  # replace with your actual file

# ---------------------------
# 1. Data Preprocessing
# ---------------------------

# Encode categorical features
for col in df.select_dtypes(include='object').columns:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))

# Define X and y
X = df.drop('Churn', axis=1)  # replace 'Churn' with your target column
y = df['Churn']

# ---------------------------
# 2. Train-Test Split
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------------------
# 3. Feature Scaling
# ---------------------------

# Standardization
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

# OR: Normalization
minmax_scaler = MinMaxScaler()
X_train_norm = minmax_scaler.fit_transform(X_train)
X_test_norm = minmax_scaler.transform(X_test)

# ---------------------------
# 4. Dimensionality Reduction with PCA
# ---------------------------

# Apply PCA after Standard Scaling
pca = PCA(n_components=5)  # change the number of components as needed
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)

print(f"Explained Variance Ratio (PCA): {pca.explained_variance_ratio_}")

# ---------------------------
# 5. Feature Selection
# ---------------------------

# Using chi2 (for non-negative features → use normalized data)
chi2_selector = SelectKBest(score_func=chi2, k=5)
X_train_chi2 = chi2_selector.fit_transform(X_train_norm, y_train)
X_test_chi2 = chi2_selector.transform(X_test_norm)

print("Top features (chi2):", X.columns[chi2_selector.get_support()])

# Using f_classif (for standardized or raw continuous features)
f_selector = SelectKBest(score_func=f_classif, k=5)
X_train_f = f_selector.fit_transform(X_train_std, y_train)
X_test_f = f_selector.transform(X_test_std)

print("Top features (f_classif):", X.columns[f_selector.get_support()])
