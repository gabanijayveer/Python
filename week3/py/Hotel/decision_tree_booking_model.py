import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load data
df = pd.read_csv('hotel_booking_data1.csv')

# Encode target variable
df['booking_status'] = df['booking_status'].map({'Not_Canceled': 0, 'Canceled': 1})

# Create new feature: total_nights
df['total_nights'] = df['no_of_week_nights'] + df['no_of_weekend_nights']

# Drop unused columns
df.drop(columns=['Booking_ID', 'arrival_year', 'arrival_date'], inplace=True)

# Separate features and target
X = df.drop(columns=['booking_status'])
y = df['booking_status']

# Identify categorical columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

# Build preprocessing and model pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first'), categorical_cols)
    ],
    remainder='passthrough'
)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier(random_state=42))
])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
pipeline.fit(X_train, y_train)

# Predict
y_pred = pipeline.predict(X_test)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save model
with open('decision_tree_model.pkl', 'wb') as f:
    pickle.dump(pipeline, f)

print("Model saved as decision_tree_model.pkl")
