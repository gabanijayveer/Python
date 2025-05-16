import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from datetime import datetime
# Load dataset
data = pd.read_csv("diabetes_data.csv")

# 🔧 Fix column names by stripping whitespace
data.columns = data.columns.str.strip()

# Convert 'last check up date' to days since last checkup
data['last check up date'] = pd.to_datetime(data['last check up date'])
data['days_since_checkup'] = (datetime.today() - data['last check up date']).dt.days

# Drop Name and date column
data = data.drop(columns=['Name', 'last check up date'])

# Encode target
label_encoder = LabelEncoder()
data['result'] = label_encoder.fit_transform(data['result'])

# Feature-target split
X = data.drop(columns='result')
y = data['result']

# Define columns
categorical_features = ['Gender', 'Blood Type']
numerical_features = ['Age', 'FBS', 'OGTT', 'HbA1c', 'RBS', 'days_since_checkup']

# Preprocessing
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# ColumnTransformer
preprocessor = ColumnTransformer(transformers=[
    ('num', numerical_transformer, numerical_features),
    ('cat', categorical_transformer, categorical_features)
])

# Final pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000))
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Fit pipeline
pipeline.fit(X_train, y_train)

# Predict
predictions = pipeline.predict(X_test)
print("Predicted Classes:", label_encoder.inverse_transform(predictions))
