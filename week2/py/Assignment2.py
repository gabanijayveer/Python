import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Step 1: Load the dataset

df = pd.read_csv('healthcare_dataset.csv')

print("Initial Data:")
print(df.head())

print("\nData Info:")
print(df.info())
# Step 2: Data Cleaning


print("\nMissing values before cleaning:")
print(df.isnull().sum())

# Handle missing values

df['Age'].fillna(df['Age'].mean(), inplace=True)  # Fill missing Age with mean
df['Billing Amount'].fillna(df['Billing Amount'].mean(), inplace=True)  # Fill missing Billing Amount with mean

# Convert relevant columns to appropriate data types if necessary
df['Date of Admission'] = pd.to_datetime(df['Date of Admission'], errors='coerce')
df['Discharge Date'] = pd.to_datetime(df['Discharge Date'], errors='coerce')
# Drop any remaining rows with NaN values
df.dropna(inplace=True)

# Step 3: Descriptive Statistics
# Get descriptive statistics for numerical columns
print("\nDescriptive Statistics:")
print(df.describe())
# Get counts of unique values in categorical columns
print("\nMedical Condition Counts:")
print(df['Medical Condition'].value_counts())
print("\nGender Counts:")
print(df['Gender'].value_counts())
# Step 4: Visualization
# Filter for diabetes patients
diabetes_df = df[df['Medical Condition'] == 'Diabetes']
# 1. Age Distribution of Diabetes Patients
plt.figure(figsize=(8, 6))
plt.hist(diabetes_df['Age'], bins=20, color='skyblue', edgecolor='black')
plt.title('Age Distribution of Diabetes Patients')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()
# 2. Average Billing Amount by Gender
avg_billing_gender = diabetes_df.groupby('Gender')['Billing Amount'].mean().reset_index()
plt.figure(figsize=(8, 6))
plt.bar(avg_billing_gender['Gender'], avg_billing_gender['Billing Amount'], color='lightgreen', edgecolor='black')
plt.title('Average Billing Amount for Diabetes Patients by Gender')
plt.xlabel('Gender')
plt.ylabel('Average Billing Amount')
plt.grid(axis='y')
plt.show()
# 3. Correlation Matrix
correlation_matrix = diabetes_df[['Age', 'Billing Amount']].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=1, linecolor='black')
plt.title('Correlation between Age and Billing Amount')
plt.show()
# 4. Average Billing Amount by Blood Type
avg_billing_blood_type = diabetes_df.groupby('Blood Type')['Billing Amount'].mean().reset_index()
plt.figure(figsize=(8, 6))
plt.bar(avg_billing_blood_type['Blood Type'], avg_billing_blood_type['Billing Amount'], color='coral', edgecolor='black')
plt.title('Average Billing Amount for Diabetes Patients by Blood Type')
plt.xlabel('Blood Type')
plt.ylabel('Average Billing Amount')
plt.grid(axis='y')
plt.show()


df['Is_Diabetes'] = df['Medical Condition'].apply(lambda x: 1 if x == 'Diabetes' else 0)

# Check the distribution of the new binary variable
print(df['Is_Diabetes'].value_counts())

# Compare with other features (e.g., Age, Billing Amount)
sns.boxplot(x='Is_Diabetes', y='Age', data=df)
plt.title('Age Distribution for Diabetes vs Non-Diabetes')
plt.show()

sns.boxplot(x='Is_Diabetes', y='Billing Amount', data=df)
plt.title('Billing Amount for Diabetes vs Non-Diabetes')
plt.show()
