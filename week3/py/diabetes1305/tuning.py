import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from datetime import datetime

# Load data
data = pd.read_csv("diabetes_data.csv")
data.columns = data.columns.str.strip()  # Clean column names

# Date feature engineering
data['last check up date'] = pd.to_datetime(data['last check up date'])
data['days_since_checkup'] = (datetime.today() - data['last check up date']).dt.days

# Drop unused columns
data = data.drop(columns=['Name', 'last check up date'])

# Encode target
label_encoder = LabelEncoder()
data['result'] = label_encoder.fit_transform(data['result'])

# Split features and target
X = data.drop(columns='result')
y = data['result']

# Column groups
numerical_features = ['Age', 'FBS', 'OGTT', 'HbA1c', 'RBS', 'days_since_checkup']
categorical_features = ['Gender', 'Blood Type']

# Preprocessing pipelines
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Column transformer
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numerical_features),
    ('cat', categorical_transformer, categorical_features)
])

# Full pipeline with feature selection and model
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('feature_selection', SelectKBest(score_func=f_classif)),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Parameter grid for GridSearchCV
param_grid = {
    'feature_selection__k': [4, 6, 'all'],
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [None, 5, 10]
}

# GridSearchCV
grid_search = GridSearchCV(pipeline, param_grid, cv=2, scoring='accuracy', n_jobs=-1)

print("Dataset size:", X.shape[0])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Fit
grid_search.fit(X_train, y_train)

# Evaluate
y_pred = grid_search.predict(X_test)
print("Best Parameters:", grid_search.best_params_)
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=label_encoder.classes_))

	