import pandas as pd

# Step 1: Create a dictionary with sample data
data = {
    'Name': ['John Doe', 'Jane Smith', 'Alice Johnson', 'Bob Brown', 'Eve Davis'],
    'Age': [55, 42, 60, 35, 50],
    'Gender': ['Male', 'Female', 'Female', 'Male', 'Female'],
    'Blood Type': ['O+', 'A-', 'B+', 'O-', 'AB+'],
    'Medical Condition': ['Hypertension', 'Diabetes', 'Asthma', 'Heart Disease', 'Hypertension'],
    'Date of Admission': ['2023-03-01', '2023-04-15', '2023-02-20', '2023-05-10', '2023-06-12'],
    'Doctor': ['Dr. Lee', 'Dr. Adams', 'Dr. Brown', 'Dr. Clark', 'Dr. White'],
    'Hospital': ['City Hospital', 'Sunrise Medical Center', 'Greenfield Clinic', 'Red Cross Hospital', 'Evergreen Health'],
    'Insurance Provider': ['Blue Cross', 'HealthNet', 'Aetna', 'Cigna', 'United Healthcare'],
    'Billing Amount': [1200, 1500, 800, 1000, 1300],
    'Room Number': [101, 102, 103, 104, 105],
    'Admission Type': ['Emergency', 'Elective', 'Emergency', 'Routine', 'Emergency'],
    'Discharge Date': ['2023-03-05', '2023-04-18', '2023-02-22', '2023-05-15', '2023-06-16'],
    'Medication': ['Lisinopril', 'Metformin', 'Albuterol', 'Aspirin', 'Lisinopril'],
    'Test Results': [5.6, 6.7, 3.2, 4.5, 5.2]  # Assuming these are test scores or results.
}

# Step 2: Convert the dictionary to a DataFrame
df = pd.DataFrame(data)

# Step 3: Save the DataFrame as a CSV file
df.to_csv('patient_data.csv', index=False)

print("CSV file 'patient_data.csv' has been created.")
