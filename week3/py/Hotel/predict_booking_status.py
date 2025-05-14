import pandas as pd
import pickle

# Load the trained model
with open('decision_tree_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load or prepare new data
df = pd.read_csv('hotel_booking_data1.csv')

# Optional: Add 'total_nights' again if not in CSV
df['total_nights'] = df['no_of_week_nights'] + df['no_of_weekend_nights']

# Drop unused columns
X = df.drop(columns=['Booking_ID', 'arrival_year', 'arrival_date', 'booking_status'])

# Predict
predictions = model.predict(X)

# Add predictions to DataFrame
df['Predicted_Booking_Status'] = predictions
df['Predicted_Booking_Status'] = df['Predicted_Booking_Status'].map({0: 'Not_Canceled', 1: 'Canceled'})

# Save results
df.to_csv('predicted_booking_status.csv', index=False)

print("Predictions saved to predicted_booking_status.csv")
