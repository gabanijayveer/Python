# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import LabelEncoder

# Step 1: Load the Dataset
df = pd.read_csv('Patient_Admission_Readmission.csv')

# Step 2: Explore the Data
print("Data Overview:")
print(df.head())
print("\nData Info:")
print(df.info())
print("\nData Description:")
print(df.describe())# Step 3: Data Preprocessing
# Handle Missing Values (imputing numerical and categorical columns differently)

# Impute numerical columns with the mean
numeric_columns = df.select_dtypes(include=[np.number]).columns  # Select only numeric columns
df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())

# Impute categorical columns with the mode (most frequent value)
categorical_columns = df.select_dtypes(include=[object]).columns  # Select only categorical columns
for col in categorical_columns:
    df[col].fillna(df[col].mode()[0], inplace=True)  # Mode imputation

# Check the dataset again after handling missing values
print("\nData after imputation:")
print(df.isnull().sum())  # Verify if there are any missing values left


# Encode categorical variables
# If 'Readmission' is a binary column like 'Yes/No', encode it as 1/0
if df['Readmission'].dtype == 'object':
    df['Readmission'] = df['Readmission'].map({'Yes': 1, 'No': 0})

# Apply get_dummies for categorical features
df = pd.get_dummies(df, drop_first=True)

# Check if 'Readmission' is in the dataset and set it as target variable
if 'Readmission' in df.columns:
    X = df.drop(['Readmission'], axis=1)  # Features
    y = df['Readmission']  # Target variable
else:
    print("The 'Readmission' column is missing from the dataset.")
    exit()

# Split the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Supervised Learning
# Logistic Regression
print("\n--- Logistic Regression ---")
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

# Confusion Matrix and Classification Report
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_lr))
print("Classification Report:")
print(classification_report(y_test, y_pred_lr))

# Decision Tree Classifier
print("\n--- Decision Tree Classifier ---")
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_dt))
print("Classification Report:")
print(classification_report(y_test, y_pred_dt))

# Step 5: Unsupervised Learning
# Clustering (KMeans)
print("\n--- KMeans Clustering ---")
kmeans = KMeans(n_clusters=2, random_state=42)  # Choose number of clusters
kmeans.fit(X)
df['Cluster'] = kmeans.labels_

# Hierarchical Clustering
print("\n--- Hierarchical Clustering Dendrogram ---")
linked = linkage(X, 'ward')
plt.figure(figsize=(10, 7))
dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=True)
plt.title('Hierarchical Clustering Dendrogram')
plt.show()

# Dimensionality Reduction (PCA)
print("\n--- PCA Visualization ---")
pca = PCA(n_components=2)  # Reduce to 2 dimensions
X_pca = pca.fit_transform(X)
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans.labels_, cmap='viridis')
plt.title('PCA of Healthcare Dataset')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar()
plt.show()

# Step 6: Model Evaluation
# Cross-Validation
print("\n--- Cross-Validation Scores ---")
cv_scores = cross_val_score(lr_model, X, y, cv=5)
print("Cross-Validation Scores:", cv_scores)

# Accuracy, Precision, Recall, F1 Score
accuracy = accuracy_score(y_test, y_pred_lr)
precision = precision_score(y_test, y_pred_lr)
recall = recall_score(y_test, y_pred_lr)
f1 = f1_score(y_test, y_pred_lr)
roc_auc = roc_auc_score(y_test, y_pred_lr)
print(f"\nLogistic Regression - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}, ROC AUC: {roc_auc}")


# Step 6: Model Evaluation
# Cross-Validation
print("\n--- Cross-Validation Scores ---")
cv_scores = cross_val_score(lr_model, X, y, cv=5)
print("Cross-Validation Scores:", cv_scores)

# Logistic Regression Evaluation Metrics
accuracy_lr = accuracy_score(y_test, y_pred_lr)
precision_lr = precision_score(y_test, y_pred_lr)
recall_lr = recall_score(y_test, y_pred_lr)
f1_lr = f1_score(y_test, y_pred_lr)
roc_auc_lr = roc_auc_score(y_test, y_pred_lr)

# Decision Tree Evaluation Metrics
accuracy_dt = accuracy_score(y_test, y_pred_dt)
precision_dt = precision_score(y_test, y_pred_dt)
recall_dt = recall_score(y_test, y_pred_dt)
f1_dt = f1_score(y_test, y_pred_dt)
roc_auc_dt = roc_auc_score(y_test, y_pred_dt)

# Displaying the evaluation metrics
print(f"\nLogistic Regression - Accuracy: {accuracy_lr}, Precision: {precision_lr}, Recall: {recall_lr}, F1 Score: {f1_lr}, ROC AUC: {roc_auc_lr}")
print(f"Decision Tree - Accuracy: {accuracy_dt}, Precision: {precision_dt}, Recall: {recall_dt}, F1 Score: {f1_dt}, ROC AUC: {roc_auc_dt}")

# Bar chart for evaluation metrics of both models
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']
lr_scores = [accuracy_lr, precision_lr, recall_lr, f1_lr, roc_auc_lr]
dt_scores = [accuracy_dt, precision_dt, recall_dt, f1_dt, roc_auc_dt]

# Plotting the bar chart
fig, ax = plt.subplots(figsize=(10, 6))
bar_width = 0.35
index = np.arange(len(metrics))

# Bars for Logistic Regression and Decision Tree
bar1 = ax.bar(index, lr_scores, bar_width, label='Logistic Regression', color='b')
bar2 = ax.bar(index + bar_width, dt_scores, bar_width, label='Decision Tree', color='r')

# Adding labels and titles
ax.set_xlabel('Metrics')
ax.set_ylabel('Scores')
ax.set_title('Model Comparison: Logistic Regression vs Decision Tree')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(metrics)
ax.legend()

# Show plot
plt.tight_layout()
plt.show()
