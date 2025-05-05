import pandas as pd

# Creating a mock dataset with equal length arrays (50 rows)
data = {
    'LoanAmount': [5000, 10000, 20000, 15000, 25000, 3000, 5000, 12000, 8000, 22000,
                   11000, 20000, 7000, 6000, 10000, 15000, 25000, 3000, 17000, 4000,
                   14000, 8000, 9000, 19000, 6000, 13000, 10000, 11000, 12000, 16000,
                   5000, 20000, 14000, 18000, 7000, 11000, 22000, 9000, 13000, 25000,
                   8000, 15000, 4000, 17000, 10000, 11000, 15000, 20000, 5000, 12000, 9000],
    'InterestRate': [5.5, 7.0, 6.5, 8.0, 6.0, 5.0, 7.5, 6.0, 6.8, 5.5, 
                     7.2, 6.4, 5.9, 8.1, 5.6, 7.3, 5.2, 7.0, 6.9, 6.2, 
                     6.3, 7.4, 6.1, 5.8, 7.1, 5.3, 7.5, 6.6, 5.7, 6.8, 
                     5.0, 7.6, 6.4, 5.1, 7.3, 6.0, 5.9, 7.2, 6.5, 6.4, 
                     7.1, 6.7, 6.9, 7.3, 5.6, 6.8, 7.2, 5.3, 7.0],
    'ApprovalStatus': ['Approved', 'Denied', 'Approved', 'Approved', 'Denied', 'Approved', 
                       'Denied', 'Approved', 'Approved', 'Denied', 'Approved', 'Approved', 
                       'Approved', 'Denied', 'Approved', 'Denied', 'Approved', 'Approved', 
                       'Approved', 'Denied', 'Approved', 'Denied', 'Approved', 'Denied', 
                       'Approved', 'Approved', 'Denied', 'Approved', 'Denied', 'Approved', 
                       'Denied', 'Approved', 'Approved', 'Approved', 'Denied', 'Approved', 
                       'Denied', 'Approved', 'Denied', 'Approved', 'Approved', 'Approved', 
                       'Denied', 'Approved', 'Denied', 'Approved', 'Approved', 'Denied', 
                       'Approved', 'Approved', 'Approved'],
    'Year': [2021, 2021, 2020, 2022, 2021, 2022, 2021, 2020, 2021, 2020, 
             2022, 2022, 2020, 2021, 2020, 2022, 2021, 2020, 2022, 2021, 
             2021, 2022, 2020, 2022, 2020, 2021, 2020, 2022, 2022, 2021, 
             2020, 2021, 2022, 2021, 2020, 2022, 2020, 2020, 2021, 2022, 
             2022, 2021, 2022, 2020, 2022, 2020, 2020, 2021, 2022, 2021]
}

# Create DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv("loan_data.csv", index=False)

# Display the first few rows to verify
print(df.head())
