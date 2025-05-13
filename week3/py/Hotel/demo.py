import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import random

# # Step 1: Load or create the initial data (if reading from CSV, use pd.read_csv)
# # Here, we simulate the data as an example:
# initial_data = [
#     ["BK1001", 2, 0, 1, 2, "Meal Plan 1", 0, "Room_Type 1", 34, 2024, 7, 12, "Online", 0, 0, 2, 120.75, 1, 0],
#     ["BK1002", 1, 1, 2, 3, "Meal Plan 2", 1, "Room_Type 2", 67, 2024, 8, 20, "Offline", 1, 1, 0, 150.00, 0, 1],
#     ["BK1003", 3, 2, 3, 4, "Meal Plan 1", 1, "Room_Type 3", 120, 2024, 6, 5, "Corporate", 0, 0, 3, 200.25, 2, 0],
#     ["BK1004", 2, 0, 0, 1, "Not Selected", 0, "Room_Type 1", 5, 2024, 9, 17, "Online", 0, 2, 1, 85.50, 1, 1],
#     ["BK1005", 4, 1, 2, 5, "Meal Plan 3", 1, "Room_Type 4", 90, 2024, 10, 8, "Offline", 1, 0, 0, 300.00, 3, 0],
#     ["BK1006", 2, 2, 1, 3, "Meal Plan 2", 0, "Room_Type 2", 25, 2024, 11, 25, "Corporate", 0, 1, 2, 170.45, 0, 1],
#     ["BK1007", 1, 0, 0, 2, "Not Selected", 1, "Room_Type 1", 12, 2024, 12, 30, "Online", 0, 0, 1, 99.99, 0, 0],
#     ["BK1008", 2, 1, 1, 4, "Meal Plan 1", 1, "Room_Type 5", 200, 2024, 6, 2, "Online", 0, 3, 5, 250.30, 4, 1],
#     ["BK1009", 3, 0, 2, 6, "Meal Plan 3", 0, "Room_Type 6", 180, 2024, 7, 14, "Corporate", 1, 0, 1, 190.75, 2, 0],
#     ["BK1010", 2, 2, 3, 5, "Meal Plan 2", 1, "Room_Type 3", 75, 2024, 8, 9, "Offline", 0, 1, 0, 160.00, 1, 1]
# ]

# # Create a DataFrame
# columns = [
#     "Booking_ID", "no_of_adults", "no_of_children", "no_of_weekend_nights", "no_of_week_nights",
#     "type_of_meal_plan", "required_car_parking_space", "room_type_reserved", "lead_time", "arrival_year", 
#     "arrival_month", "arrival_date", "market_segment_type", "repeated_guest", "no_of_previous_cancellations",
#     "no_of_previous_bookings_not_canceled", "avg_price_per_room", "no_of_special_requests", "booking_status"
# ]
# df = pd.DataFrame(initial_data, columns=columns)

# # Step 2: Add more synthetic data (simulating new rows)
# new_rows = []
# for i in range(100):  # Let's add 100 more rows for example
#     booking_id = f"BK{1011 + i}"
#     no_of_adults = random.choice([1, 2, 3, 4])
#     no_of_children = random.choice([0, 1, 2])
#     no_of_weekend_nights = random.choice([0, 1, 2, 3])
#     no_of_week_nights = random.choice([1, 2, 3, 4])
#     type_of_meal_plan = random.choice(['Meal Plan 1', 'Meal Plan 2', 'Meal Plan 3', 'Not Selected'])
#     required_car_parking_space = random.choice([0, 1])
#     room_type_reserved = random.choice(['Room_Type 1', 'Room_Type 2', 'Room_Type 3', 'Room_Type 4', 'Room_Type 5'])
#     lead_time = random.randint(1, 200)
#     arrival_year = random.choice([2024, 2025])
#     arrival_month = random.randint(1, 12)
#     arrival_date = random.randint(1, 31)
#     market_segment_type = random.choice(['Online', 'Offline', 'Corporate'])
#     repeated_guest = random.choice([0, 1])
#     no_of_previous_cancellations = random.randint(0, 3)
#     no_of_previous_bookings_not_canceled = random.randint(0, 5)
#     avg_price_per_room = random.uniform(50.0, 500.0)
#     no_of_special_requests = random.randint(0, 3)
#     booking_status = random.choice([0, 1])  # 0: Not Canceled, 1: Canceled
    
#     new_rows.append([
#         booking_id, no_of_adults, no_of_children, no_of_weekend_nights, no_of_week_nights, type_of_meal_plan,
#         required_car_parking_space, room_type_reserved, lead_time, arrival_year, arrival_month, arrival_date,
#         market_segment_type, repeated_guest, no_of_previous_cancellations, no_of_previous_bookings_not_canceled,
#         avg_price_per_room, no_of_special_requests, booking_status
#     ])
df = pd.read_csv("hotel_booking_data.csv")

# Create a DataFrame from new rows
new_data_df = pd.DataFrame(df, columns=df.columns)

# Append new synthetic data to existing data
df = pd.concat([df, new_data_df], ignore_index=True)

# Step 3: Preprocess data
categorical_cols = [
    "type_of_meal_plan", "room_type_reserved", "market_segment_type", "booking_status"
]
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

df.drop("Booking_ID", axis=1, inplace=True)

# Step 4: Features and Target
X = df.drop("booking_status", axis=1)
y = df["booking_status"]

# Step 5: Split dataset (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42  # 80% train, 20% test
)

# Step 6: Train Decision Tree model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Step 7: Predictions
y_pred = model.predict(X_test)

# Step 8: Evaluation
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nAccuracy Score:", accuracy_score(y_test, y_pred))

# Step 9: Visualize Decision Tree
plt.figure(figsize=(16, 8))
plot_tree(model, feature_names=X.columns, class_names=["Not_Canceled", "Canceled"], filled=True)
plt.title("Decision Tree - Booking Cancellation Prediction")
plt.show()
