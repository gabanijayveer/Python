import pandas as pd
import joblib

# Load your trained model
model = joblib.load('diabetes_model.pkl')

# Load dataset
df = pd.read_csv('expanded_diabetes_risk_dataset.csv')

# Step 1: Split Blood_Pressure into two numeric columns
df[['Systolic_BP', 'Diastolic_BP']] = df['Blood_Pressure'].str.split('/', expand=True)
df['Systolic_BP'] = pd.to_numeric(df['Systolic_BP'])
df['Diastolic_BP'] = pd.to_numeric(df['Diastolic_BP'])

# Step 2: Encode categorical features
medical_condition_map = {'PCOS': 0, 'High Blood Pressure': 1}
gestational_map = {'No': 0, 'Yes': 1}

df['Medical_Conditions'] = df['Medical_Conditions'].map(medical_condition_map)
df['Gestational_Diabetes'] = df['Gestational_Diabetes'].map(gestational_map)

# Step 3: Prepare feature columns (must match training order)
X = df[['Fasting_Blood_Glucose', 'OGTT_Result', 'Hemoglobin_A1c', 'Random_Blood_Sugar',
        'Systolic_BP', 'Diastolic_BP', 'LDL', 'HDL', 'Medical_Conditions', 'BMI', 'Gestational_Diabetes']]

# Step 4: Predict
predictions = model.predict(X)

# Step 5: Add predictions to DataFrame
df['Predicted_Diabetes_Status'] = ['Diabetes' if p == 1 else 'No Diabetes' for p in predictions]

# Step 6: Save to CSV or print results
df.to_csv('predicted_results.csv', index=False)
print(df[['Name', 'Predicted_Diabetes_Status']])
