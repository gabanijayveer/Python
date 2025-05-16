import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, f1_score
import joblib

# Load dataset
df = pd.read_csv('telco_churn.csv')

# Convert TotalCharges to numeric and handle missing values
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

# Encode target variable
df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1})

# Drop irrelevant ID column and target from features
X = df.drop(['CustomerID', 'Churn'], axis=1)
y = df['Churn']

# Identify numerical and categorical columns
num_cols = ['Tenure', 'MonthlyCharges', 'TotalCharges']
cat_cols = [col for col in X.columns if col not in num_cols]

# Train/test split with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Preprocessing transformers
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), cat_cols)
])

# Pipeline with preprocessing and classifier (with class_weight balanced)
pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('clf', RandomForestClassifier(random_state=42, class_weight='balanced'))
])

# Hyperparameter grid for tuning
param_grid = {
    'clf__n_estimators': [100, 200],
    'clf__max_depth': [None, 10, 20],
    'clf__min_samples_split': [2, 5]
}

# Stratified k-fold cross-validation
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# Grid search with multiple scoring metrics, refit based on f1
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=cv,
    scoring=['accuracy', 'f1', 'roc_auc'],
    refit='f1',
    n_jobs=-1,
    verbose=2
)

# Fit model
grid_search.fit(X_train, y_train)

# Output best parameters and best scores
print(" Best Parameters:", grid_search.best_params_)
print(f" Best CV F1 Score: {grid_search.best_score_:.4f}")

# Test set predictions
y_pred = grid_search.predict(X_test)
y_proba = grid_search.predict_proba(X_test)[:, 1]

# Test set metrics
print(f"\n Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f" Test F1 Score: {f1_score(y_test, y_pred):.4f}")
print(f" Test ROC AUC: {roc_auc_score(y_test, y_proba):.4f}")

print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))

# Save the best model to disk
joblib.dump(grid_search.best_estimator_, 'best_telco_churn_model.pkl')
print("\n💾 Model saved to 'best_telco_churn_model.pkl'")
