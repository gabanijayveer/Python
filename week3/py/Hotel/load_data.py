import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# Step 1: Load the data
df = pd.read_csv("hotel_booking_data.csv")

# Step 2: Preprocessing
# Encode categorical features
categorical_cols = [
    "type_of_meal_plan", "room_type_reserved", "market_segment_type", "booking_status"
]
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Drop Booking_ID (non-informative)
df.drop("Booking_ID", axis=1, inplace=True)

# Step 3: Features and Target
X = df.drop("booking_status", axis=1)
y = df["booking_status"]

# Step 4: Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Step 5: Train Decision Tree model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Step 6: Predictions
y_pred = model.predict(X_test)

# Step 7: Evaluation
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nAccuracy Score:", accuracy_score(y_test, y_pred))

# Step 8: Visualize Decision Tree
plt.figure(figsize=(16, 8))
plot_tree(model, feature_names=X.columns, class_names=["Not_Canceled", "Canceled"], filled=True)
plt.title("Decision Tree - Booking Cancellation Prediction")
plt.show()
