import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv('loan_data.csv')

# Encode categorical variables
label_encoders = {}
for col in ['Gender', 'Married', 'Education', 'Loan_Status']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Features and label
X = df[['Gender', 'Married', 'Education', 'ApplicantIncome', 'Credit_History', 'LoanAmount', 'Loan_Amount_Term']]
y = df['Loan_Status']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model and encoders
joblib.dump(model, 'loan_model.pkl')
joblib.dump(label_encoders, 'label_encoders.pkl')

print("✅ Model and encoders saved!")
