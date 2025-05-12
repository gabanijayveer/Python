from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model and label encoders
model = joblib.load('loan_model.pkl')
label_encoders = joblib.load('label_encoders.pkl')

@app.route('/')
def index():
    return render_template('loan_form.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form
    gender = label_encoders['Gender'].transform([data['gender']])[0]
    married = label_encoders['Married'].transform([data['married']])[0]
    education = label_encoders['Education'].transform([data['education']])[0]
    income = float(data['income'])
    credit_history = int(data['credit_history'])
    loan_amount = float(data['loan_amount'])
    loan_term = float(data['loan_term'])

    input_data = np.array([[gender, married, education, income, credit_history, loan_amount, loan_term]])

    # Predict the loan approval status and probabilities
    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data)[0]

    # Get the human-readable loan status prediction
    result = label_encoders['Loan_Status'].inverse_transform([prediction])[0]

    # Map labels to their indices to get correct probabilities
    class_to_index = {label: idx for idx, label in enumerate(label_encoders['Loan_Status'].classes_)}

    approved_index = class_to_index.get('Approved')
    not_approved_index = class_to_index.get('Not Approved')

    approved_proba = prediction_proba[approved_index] if approved_index is not None else 0.0
    not_approved_proba = prediction_proba[not_approved_index] if not_approved_index is not None else 0.0

    # Create a result message showing both probabilities
    prediction_message = (
        f"Loan Prediction: {result}<br>"
        f"Probability of Loan Approved: {approved_proba * 100:.2f}%<br>"
        f"Probability of Loan Not Approved: {not_approved_proba * 100:.2f}%"
    )

    return render_template('loan_form.html', prediction=prediction_message)

if __name__ == '__main__':
    app.run(debug=True)
