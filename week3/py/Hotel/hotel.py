# hospitality_booking_cancellation.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the dataset
df = pd.read_csv('hotel_booking_data.csv')

# Data Preprocessing
# Encode categorical columns
label_encoder = LabelEncoder()
df['type_of_meal_plan'] = label_encoder.fit_transform(df['type_of_meal_plan'])
df['room_type_reserved'] = label_encoder.fit_transform(df['room_type_reserved'])
df['market_segment_type'] = label_encoder.fit_transform(df['market_segment_type'])
df['booking_status'] = label_encoder.fit_transform(df['booking_status'])

# Combine arrival date columns (if necessary)
# df['arrival_date'] = pd.to_datetime(df[['arrival_year', 'arrival_month', 'arrival_date']])

# Split data into features and target
X = df.drop(columns=['booking_status', 'Booking_ID'])  # Drop irrelevant columns
y = df['booking_status']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Decision Tree model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
