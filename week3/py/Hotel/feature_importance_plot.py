import pandas as pd
import pickle
import matplotlib.pyplot as plt

# Load model
with open('decision_tree_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load data to get feature names
df = pd.read_csv('hotel_booking_data1.csv')
df['total_nights'] = df['no_of_week_nights'] + df['no_of_weekend_nights']
df.drop(columns=['Booking_ID', 'arrival_year', 'arrival_date', 'booking_status'], inplace=True)

# Get preprocessor from pipeline
ohe = model.named_steps['preprocessor'].named_transformers_['cat']
categorical_features = ohe.get_feature_names_out(df.select_dtypes(include='object').columns)
numerical_features = df.select_dtypes(exclude='object').columns.tolist()
feature_names = list(categorical_features) + numerical_features

# Get feature importance
importances = model.named_steps['classifier'].feature_importances_

# Create a DataFrame for visualization
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# Plot
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
plt.xlabel("Feature Importance")
plt.title("Decision Tree Feature Importance")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.show()
