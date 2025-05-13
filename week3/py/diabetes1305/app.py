from flask import Flask, render_template, request
import joblib
import numpy as np

# Load model and label encoder
model = joblib.load('diabetes_risk.pkl')
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
        fasting_glucose = safe_float(request.form.get('FBS'))
        ogtt = safe_float(request.form.get('OGTT'))
        hba1c = safe_float(request.form.get('HbA1c'))
        random_sugar = safe_float(request.form.get('RBS'))

        # Prepare data for prediction (only 4 features)
        input_data = np.array([[fasting_glucose, ogtt, hba1c, random_sugar]])

        prediction = model.predict(input_data)
        predicted_label = label_encoder.inverse_transform(prediction)[0]

        result = f"🩺 Prediction: {predicted_label}"

    except Exception as e:
        result = f"❌ Error: {str(e)}"

    return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
