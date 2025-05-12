import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

# 1. Load dataset
df = pd.read_csv("expanded_diabetes_risk_dataset.csv")

# 2. Preprocess blood pressure into systolic/diastolic
df[['Systolic_BP', 'Diastolic_BP']] = df['Blood_Pressure'].str.split('/', expand=True).astype(int)
df.drop(columns=['Name', 'Blood_Pressure'], inplace=True)

# 3. Encode categorical columns
label_encoders = {}
for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# 4. Define features and target
X = df.drop("Diabetes_Status", axis=1)
y = df["Diabetes_Status"]

# 5. Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Train model with class weight to handle imbalance
model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

# 7. Make predictions
y_pred = model.predict(X_test)

# 8. Evaluate
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, zero_division=0))
print("✅ Accuracy:", accuracy_score(y_test, y_pred))

# 9. Feature importance
importances = model.feature_importances_
features = X.columns
feat_df = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values(by="Importance", ascending=False)

# 10. Plot feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x="Importance", y="Feature", data=feat_df, palette="Blues_d", legend=False)
plt.title("Feature Importance in Diabetes Risk Prediction")
plt.tight_layout()
plt.show()
