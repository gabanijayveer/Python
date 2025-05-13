from flask import Flask, render_template
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained model and dataset
model = joblib.load('diabetes_model.pkl')
df = pd.read_csv('expanded_diabetes_risk_dataset1.csv')

# Preprocess dataset
def preprocess(df):
    # Split blood pressure
    df[['Systolic_BP', 'Diastolic_BP']] = df['Blood_Pressure'].str.split('/', expand=True)
    df['Systolic_BP'] = pd.to_numeric(df['Systolic_BP'], errors='coerce')
    df['Diastolic_BP'] = pd.to_numeric(df['Diastolic_BP'], errors='coerce')

    # Encode categorical features
    medical_condition_map = {'PCOS': 0, 'High Blood Pressure': 1}
    gestational_map = {'No': 0, 'Yes': 1}

    df['Medical_Conditions'] = df['Medical_Conditions'].map(medical_condition_map)
    df['Gestational_Diabetes'] = df['Gestational_Diabetes'].map(gestational_map)

    # Fill any missing or invalid data
    df.fillna(0, inplace=True)

    return df

df = preprocess(df)

# Select features used for prediction
features = ['Fasting_Blood_Glucose', 'OGTT_Result', 'Hemoglobin_A1c', 'Random_Blood_Sugar',
            'Systolic_BP', 'Diastolic_BP', 'LDL', 'HDL', 'Medical_Conditions', 'BMI', 'Gestational_Diabetes']

# Predict
df['Predicted_Diabetes_Status'] = model.predict(df[features])
df['Predicted_Diabetes_Status'] = df['Predicted_Diabetes_Status'].map({0: 'No Diabetes', 1: 'Diabetes'})

# Limit to first 500 records
df_results = df[['Name', 'Fasting_Blood_Glucose', 'OGTT_Result', 'Hemoglobin_A1c',
                 'Random_Blood_Sugar', 'BMI', 'Predicted_Diabetes_Status']].head(500)

@app.route('/')
def index():
    return render_template('results.html', results=df_results.to_dict(orient='records'))

if __name__ == '__main__':
    app.run(debug=True)
# from flask import Flask, render_template
# import pandas as pd
# import joblib

# app = Flask(__name__)

# # Load the trained model
# model = joblib.load('diabetes_model.pkl')

# # Load dataset
# df = pd.read_csv('diabetes_data.csv')

# # Preprocess function
# def preprocess(df):
#     df = df[['Name', 'Number of Pregnancies', 'BMI', 'Insulin Level', 'Outcome']]
#     df.dropna(inplace=True)
#     return df

# # Preprocess data
# df = preprocess(df)

# # Features used for prediction
# features = ['Number of Pregnancies', 'BMI', 'Insulin Level']

# # Make predictions
# df['Predicted_Diabetes_Status'] = model.predict(df[features])
# df['Predicted_Diabetes_Status'] = df['Predicted_Diabetes_Status'].map({0: 'No Diabetes', 1: 'Diabetes'})

# # Limit to 500 records
# df_results = df[['Name', 'Number of Pregnancies', 'BMI', 'Insulin Level', 'Predicted_Diabetes_Status']].head(500)

# # Count summary
# count_summary = df_results['Predicted_Diabetes_Status'].value_counts().to_dict()

# @app.route('/')
# def index():
#     return render_template('results.html', results=df_results.to_dict(orient='records'), summary=count_summary)

# if __name__ == '__main__':
#     app.run(debug=True)
