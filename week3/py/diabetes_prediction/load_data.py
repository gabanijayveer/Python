import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

# Load dataset
df = pd.read_csv('expanded_diabetes_risk_dataset1.csv')

# Split blood pressure into systolic/diastolic
df[['Systolic_BP', 'Diastolic_BP']] = df['Blood_Pressure'].str.split('/', expand=True)
df['Systolic_BP'] = pd.to_numeric(df['Systolic_BP'], errors='coerce')
df['Diastolic_BP'] = pd.to_numeric(df['Diastolic_BP'], errors='coerce')

# Encode categorical features
medical_map = {'PCOS': 0, 'High Blood Pressure': 1}
gestational_map = {'No': 0, 'Yes': 1}

df['Medical_Conditions'] = df['Medical_Conditions'].map(medical_map).fillna(0)
df['Gestational_Diabetes'] = df['Gestational_Diabetes'].map(gestational_map).fillna(0)

# Encode target
le = LabelEncoder()
df['Diabetes_Status'] = le.fit_transform(df['Diabetes_Status'])

# Save the label encoder for future use
joblib.dump(le, 'label_encoder.pkl')

# Drop rows with missing features
df.dropna(subset=[
    'Fasting_Blood_Glucose', 'OGTT_Result', 'Hemoglobin_A1c', 'Random_Blood_Sugar',
    'Systolic_BP', 'Diastolic_BP', 'LDL', 'HDL', 'Medical_Conditions',
    'BMI', 'Gestational_Diabetes'
], inplace=True)

# Define features and target
features = [
    'Fasting_Blood_Glucose', 'OGTT_Result', 'Hemoglobin_A1c', 'Random_Blood_Sugar',
    'Systolic_BP', 'Diastolic_BP', 'LDL', 'HDL',
    'Medical_Conditions', 'BMI', 'Gestational_Diabetes'
]
X = df[features]
y = df['Diabetes_Status']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

# Save model
joblib.dump(model, 'diabetes_model.pkl')

# Evaluation
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print("Model and label encoder saved.")
