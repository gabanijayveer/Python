import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Load the dataset
df = pd.read_csv('hotel_booking_data1.csv')

# Preprocess the data (Create a new feature for total nights)
df['total_nights'] = df['no_of_week_nights'] + df['no_of_weekend_nights']
df.drop(columns=['Booking_ID', 'arrival_year', 'arrival_date','type_of_meal_plan','room_type_reserved','market_segment_type'], inplace=True)

# Encode the target variable
df['booking_status'] = df['booking_status'].map({'Not_Canceled': 0, 'Canceled': 1})

# Separate features and target
X = df.drop(columns=['booking_status'])
y = df['booking_status']

# Identify categorical columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

# Handle missing values in categorical and numerical columns (imputation)
preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='mean'), X.select_dtypes(include=['float64', 'int64']).columns),  # Impute missing values in numerical columns
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),  # Impute missing categorical values with the most frequent value
            ('encoder', OneHotEncoder(drop='first'))  # Apply OneHotEncoding and drop the first category to avoid the dummy variable trap
        ]), categorical_cols)  # Apply preprocessing to categorical columns
    ]
)

# Create a pipeline with preprocessing steps and a decision tree classifier
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier(random_state=42))
])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
pipeline.fit(X_train, y_train)

# Get the trained model's decision tree classifier
dt_model = pipeline.named_steps['classifier']

# Predict the booking cancellations on the test set
y_pred = dt_model.predict(X_test)

# Create a bar chart showing the predicted vs actual cancellations
plt.figure(figsize=(8, 6))
plt.bar(['Not Canceled', 'Canceled'], [sum(y_pred == 0), sum(y_pred == 1)], color=['green', 'red'])
plt.title('Prediction of Hotel Booking Cancellations')
plt.xlabel('Booking Status')
plt.ylabel('Number of Predictions')
plt.show()
