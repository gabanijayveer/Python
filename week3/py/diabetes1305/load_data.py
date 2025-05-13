import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

# Load dataset
df = pd.read_csv('diabetes_data.csv')

# Strip whitespace from column names
df.columns = df.columns.str.strip()

# Encode target
le = LabelEncoder()
df['result'] = le.fit_transform(df['result'])

# Save the label encoder for future use
joblib.dump(le, 'label_encoder.pkl')

# Define features and target
features = ['FBS', 'OGTT', 'HbA1c', 'RBS']
X = df[features]
y = df['result']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

# Save model
joblib.dump(model, 'diabetes_risk.pkl')

# Evaluation
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print("Model and label encoder saved.")
