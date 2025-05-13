import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load dataset
df = pd.read_csv("diabetes_data.csv")

# Strip whitespace from column names
df.columns = df.columns.str.strip()

# Map 'result' to binary: 1 for Diabetes, 0 for Non-Diabetes
df['result'] = df['result'].str.strip()
df['diabetes_risk'] = df['result'].apply(lambda x: 1 if x == 'Diabetes' else 0)

# Drop unnecessary columns
df = df.drop(columns=['Name', 'last check up date', 'result'])

# Encode categorical variables
categorical_cols = ['Gender', 'Blood Type']
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Features and target
X = df.drop(columns=['diabetes_risk'])
y = df['diabetes_risk']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred,  zero_division=0))

# Feature importance
import matplotlib.pyplot as plt

feature_importance = model.feature_importances_
features = X.columns

plt.figure(figsize=(8, 5))
plt.barh(features, feature_importance)
plt.title("Feature Importance")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()
