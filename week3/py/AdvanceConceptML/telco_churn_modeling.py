# telco_churn_modeling.py

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, roc_auc_score

def main():
    # Load dataset
    df = pd.read_csv('telco_churn.csv')  # Replace with your dataset path
    
    # Encode categorical features
    for col in df.select_dtypes(include='object').columns:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    
    # Define features and target
    X = df.drop('Churn', axis=1)  # Replace 'Churn' with your target column name
    y = df['Churn']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("Training class distribution:")
    print(y_train.value_counts())
    
    # Adjust n_splits for StratifiedKFold dynamically
    min_class_count = y_train.value_counts().min()
    n_splits = min(5, min_class_count)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    print(f"Using StratifiedKFold with n_splits={n_splits} due to class distribution.")
    
    # Feature scaling
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)
    
    # Feature selection with f_classif
    selector = SelectKBest(score_func=f_classif, k=10)  # Select top 10 features
    X_train_sel = selector.fit_transform(X_train_std, y_train)
    X_test_sel = selector.transform(X_test_std)
    selected_features = X.columns[selector.get_support()]
    print("Selected features:", list(selected_features))
    
    # Models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier(random_state=42),
        'KNN': KNeighborsClassifier(),
        'SVM': SVC(probability=True),
        'Naive Bayes': GaussianNB()
    }
    
    # Cross-validation results
    print("\nCross-validation results:")
    for name, model in models.items():
        scores = cross_val_score(model, X_train_sel, y_train, cv=cv, scoring='accuracy')
        print(f"{name}: {scores.mean():.4f} ± {scores.std():.4f}")
    
    # Train and evaluate on test set
    print("\nTest set evaluation:")
    for name, model in models.items():
        model.fit(X_train_sel, y_train)
        y_pred = model.predict(X_test_sel)
        print(f"\n{name}:")
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
    
    # ROC Curve plot
    plt.figure(figsize=(8,6))
    for name, model in models.items():
        model.fit(X_train_sel, y_train)
        y_proba = model.predict_proba(X_test_sel)[:,1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.2f})")
    
    plt.plot([0,1], [0,1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Hyperparameter tuning for Random Forest
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train_sel, y_train)
    
    print("\nBest Random Forest parameters:", grid_search.best_params_)
    print(f"Best cross-validation accuracy: {grid_search.best_score_:.4f}")
    
    best_rf = grid_search.best_estimator_
    y_pred_rf = best_rf.predict(X_test_sel)
    print(f"Test set accuracy with best RF: {accuracy_score(y_test, y_pred_rf):.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_rf))
    print("Classification Report:")
    print(classification_report(y_test, y_pred_rf))
    

if __name__ == "__main__":
    main()
