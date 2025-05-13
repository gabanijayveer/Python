# src/logistic_regression_model.py
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import os

# Load dataset
df = pd.read_csv('loan_data1.csv')

# Features and target
X = df.drop('Defaulted', axis=1)
y = df['Defaulted']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
cv_scores = cross_val_score(model, X, y, cv=5)

# Save results
os.makedirs('outputs', exist_ok=True)
with open('outputs/evaluation_report.txt', 'w') as f:
    f.write("Confusion Matrix:\n")
    f.write(str(conf_matrix) + '\n\n')
    f.write("Classification Report:\n")
    f.write(class_report + '\n')
    f.write(f"Cross-validation Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})\n")

print("Model evaluation report saved to outputs/evaluation_report.txt")
