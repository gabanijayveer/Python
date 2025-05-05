
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("loan_data.csv")

# Step 1: Data Cleaning

print("Missing values before cleaning:")
print(df.isnull().sum())

df['LoanAmount'].fillna(df['LoanAmount'].mean(), inplace=True)  
df['InterestRate'].fillna(df['InterestRate'].mean(), inplace=True)  

df['ApprovalStatus'].fillna(df['ApprovalStatus'].mode()[0], inplace=True)  

df['LoanAmount'] = pd.to_numeric(df['LoanAmount'], errors='coerce') 
df['InterestRate'] = pd.to_numeric(df['InterestRate'], errors='coerce') 
df.dropna(inplace=True)

df = df.drop_duplicates()

print("\nCleaned dataset:")
print(df.head())

# Step 2: Data Analysis and Visualization

# 1. Loan Amount Distribution
plt.figure(figsize=(10, 6))
plt.hist(df['LoanAmount'], bins=50, color='skyblue', edgecolor='black')
plt.title('Distribution of Loan Amounts')
plt.xlabel('Loan Amount')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# 2. Interest Rate Distribution
plt.figure(figsize=(10, 6))
plt.hist(df['InterestRate'], bins=50, color='lightgreen', edgecolor='black')
plt.title('Distribution of Interest Rates')
plt.xlabel('Interest Rate')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# 3. Loan Approval Status
approval_status_count = df['ApprovalStatus'].value_counts()

plt.figure(figsize=(8, 6))
approval_status_count.plot(kind='bar', color='coral', edgecolor='black')
plt.title('Loan Approval Status')
plt.xlabel('Approval Status')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.grid(True)
plt.show()

# 4. Loan Amount vs Interest Rate
plt.figure(figsize=(10, 6))
plt.scatter(df['LoanAmount'], df['InterestRate'], alpha=0.5, color='purple')
plt.title('Loan Amount vs. Interest Rate')
plt.xlabel('Loan Amount')
plt.ylabel('Interest Rate')
plt.grid(True)
plt.show()

# 5. Correlation Matrix
correlation_matrix = df[['LoanAmount', 'InterestRate']].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=1, linecolor='black')
plt.title('Correlation between Loan Amount and Interest Rate')
plt.show()

# Step 3: Trends Over Time (If applicable)

# Check if there's a 'Year' column in the dataset
if 'Year' in df.columns:
    # Convert 'Year' column to datetime if it's not already
    df['Year'] = pd.to_datetime(df['Year'], format='%Y')

    # Grouping by Year and calculating the mean loan amount and approval count
    yearly_summary = df.groupby(df['Year'].dt.year)['LoanAmount'].mean().reset_index()

    # Plotting loan amount trends over time
    plt.figure(figsize=(10, 6))
    plt.plot(yearly_summary['Year'], yearly_summary['LoanAmount'], marker='o', linestyle='-', color='blue')
    plt.title('Average Loan Amount Over Time')
    plt.xlabel('Year')
    plt.ylabel('Average Loan Amount')
    plt.grid(True)
    plt.show()

    # If ApprovalStatus exists, calculate approval trends over time
    approval_by_year = df.groupby([df['Year'].dt.year, 'ApprovalStatus']).size().unstack().fillna(0)

    # Plotting approval trends over time
    approval_by_year.plot(kind='line', figsize=(12, 8))
    plt.title('Loan Approval Status Over Time')
    plt.xlabel('Year')
    plt.ylabel('Count')
    plt.legend(title='Approval Status')
    plt.grid(True)
    plt.show()

else:
    print("No 'Year' column found. Skipping time-based analysis.")
