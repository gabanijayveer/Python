import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv("diabetes_data.csv")

# Preprocessing: Encode categorical variables
label_encoder = LabelEncoder()
data['Gender'] = label_encoder.fit_transform(data['Gender'])
data['Blood Type'] = label_encoder.fit_transform(data['Blood Type'])
data['Medical Condition'] = label_encoder.fit_transform(data['Medical Condition'])
data['Admission Type'] = label_encoder.fit_transform(data['Admission Type'])
data['Medication'] = label_encoder.fit_transform(data['Medication'])
data['Doctor'] = label_encoder.fit_transform(data['Doctor'])
data['Hospital'] = label_encoder.fit_transform(data['Hospital'])
data['Insurance Provider'] = label_encoder.fit_transform(data['Insurance Provider'])

# Select features (independent variables) and target (dependent variable)
X = data[['Age', 'Gender', 'Blood Type', 'Medical Condition', 'Billing Amount', 
          'Room Number', 'Admission Type', 'Test Results', 'Number of Pregnancies', 
          'BMI', 'Insulin Level']]
y = data['Outcome']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print(classification_report(y_test, y_pred))

# Predict for individual patients
predictions = model.predict(X)
data['Predicted Outcome'] = predictions

# Show predictions with the original data
print(data[['Name', 'Predicted Outcome']])

# Build a bar chart to visualize the number of high and low risk patients
outcome_counts = data['Predicted Outcome'].value_counts()

# Plotting the bar chart
plt.figure(figsize=(8, 6))
outcome_counts.plot(kind='bar', color=['green', 'red'], alpha=0.7)
plt.title('Predicted Diabetes Risk Outcomes')
plt.xlabel('Diabetes Risk (0 = Low, 1 = High)')
plt.ylabel('Number of Patients')	
plt.xticks(rotation=0)
plt.show()
