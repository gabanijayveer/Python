import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import LabelEncoder

# Step 1: Load the Dataset
# Load the dataset (replace with your actual file path)
df = pd.read_csv('patient_data.csv')

# Display the first few rows of the dataset
print("Initial Dataset:")
print(df.head())

# Step 2: Data Preprocessing
# Handle Missing Values
# Impute numerical columns with the mean
numeric_columns = df.select_dtypes(include=[np.number]).columns
df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())

# Impute categorical columns with the mode (most frequent value)
categorical_columns = df.select_dtypes(include=[object]).columns
for col in categorical_columns:
    df[col].fillna(df[col].mode()[0], inplace=True)

# Convert Categorical Variables
label_encoder = LabelEncoder()

# Encoding 'Gender', 'Blood Type', and 'Medical Condition' for simplicity
df['Gender'] = label_encoder.fit_transform(df['Gender'])
df['Blood Type'] = label_encoder.fit_transform(df['Blood Type'])
df['Medical Condition'] = label_encoder.fit_transform(df['Medical Condition'])

# Drop Irrelevant Columns
df = df.drop(columns=['Name', 'Doctor', 'Hospital', 'Insurance Provider', 'Room Number'])

# Step 3: Feature Engineering
# Extract relevant features (e.g., Age, Medical Condition, Test Results, etc.)
# Assuming the target variable is 'Diabetes Risk', which is binary (1 = high risk, 0 = low risk)
# If the target column doesn't exist, you'd need to define it based on your dataset
if 'Diabetes Risk' not in df.columns:
    print("Creating 'Diabetes Risk' column for demonstration purposes.")
    # For demonstration purposes, assume diabetes risk based on test results (arbitrary threshold)
    df['Diabetes Risk'] = np.where(df['Test Results'] > 6, 1, 0)

# Split data into features (X) and target variable (y)
X = df.drop(columns=['Diabetes Risk'])  # Features
y = df['Diabetes Risk']  # Target variable (binary)

# Step 4: Model Training
# Split the dataset into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Logistic Regression model
model = LogisticRegression(max_iter=1000)

# Train the model
model.fit(X_train, y_train)

# Step 5: Model Evaluation
# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model using various metrics
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Classification report (precision, recall, f1-score)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ROC AUC score
roc_auc = roc_auc_score(y_test, y_pred)
print(f"\nROC AUC Score: {roc_auc:.2f}")

# Step 6: Prediction for New Patient
# Example new patient data (Age, Gender, Blood Type, Medical Condition, Test Results, etc.)
new_patient = pd.DataFrame({
    'Age': [55],
    'Gender': [1],  # Assuming 1 represents Male, 0 represents Female
    'Blood Type': [2],  # Assuming Blood Type 'B' is encoded as 2
    'Medical Condition': [1],  # Assuming encoded value for a condition
    'Test Results': [7.5]  # Assuming the test result is numeric
})

# Predict diabetes risk for the new patient
new_patient_risk = model.predict(new_patient)
print(f"Predicted Diabetes Risk (1 = High risk, 0 = Low risk): {new_patient_risk[0]}")
