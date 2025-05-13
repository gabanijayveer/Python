from flask import Flask, render_template, request
import joblib
import numpy as np

# Load model and label encoder
model = joblib.load('diabetes_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

def safe_float(value, default=0.0):
    try:
        return float(value)
    except:
        return default

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form values
        fasting_glucose = safe_float(request.form.get('fasting_glucose'))
        ogtt = safe_float(request.form.get('ogtt'))
        hba1c = safe_float(request.form.get('hba1c'))
        random_sugar = safe_float(request.form.get('random_sugar'))
        blood_pressure = request.form.get('blood_pressure', '0/0')

        if '/' not in blood_pressure:
            raise ValueError("Blood pressure must be in format like 120/80")

        systolic, diastolic = map(int, blood_pressure.split('/'))
        ldl = safe_float(request.form.get('ldl'))
        hdl = safe_float(request.form.get('hdl'))
        bmi = safe_float(request.form.get('bmi'))

        # Encode categorical features
        medical_map = {'PCOS': 0, 'High Blood Pressure': 1}
        gestational_map = {'No': 0, 'Yes': 1}
        medical = medical_map.get(request.form.get('medical_condition'), 0)
        gestational = gestational_map.get(request.form.get('gestational'), 0)

        # Prepare data for prediction
        input_data = np.array([[fasting_glucose, ogtt, hba1c, random_sugar,
                                systolic, diastolic, ldl, hdl,
                                medical, bmi, gestational]])

        prediction = model.predict(input_data)
        predicted_label = label_encoder.inverse_transform(prediction)[0]

        result = f"🩺 Prediction: {predicted_label}"

    except Exception as e:
        result = f"❌ Error: {str(e)}"

    return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
