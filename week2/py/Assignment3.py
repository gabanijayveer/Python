import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('hotel_bookings.csv')

print(df.head())


# Check for missing values
print("Missing values in the dataset:")
print(df.isnull().sum())

# Handle missing values (filling or dropping)
# For simplicity, we'll drop rows with missing target variable 'is_canceled'
df.dropna(subset=['is_canceled'], inplace=True)

# Fill or drop missing values in other columns
df['children'].fillna(0, inplace=True)  # Fill missing children with 0
df['agent'].fillna(df['agent'].mode()[0], inplace=True)  # Fill missing agent with mode
df['company'].fillna(0, inplace=True)  # Fill missing company with 0

# Convert 'arrival_date_year', 'arrival_date_month', etc. to appropriate data types
df['arrival_date_year'] = df['arrival_date_year'].astype(int)
df['arrival_date_month'] = df['arrival_date_month'].astype(str)

# Convert 'reservation_status_date' to datetime
df['reservation_status_date'] = pd.to_datetime(df['reservation_status_date'])

print(df.head())
print(df.describe())

print(df['is_canceled'].value_counts())
print(df['arrival_date_month'].value_counts())

# Count plot of cancellations vs non-cancellations

plt.figure(figsize=(8, 6))
sns.countplot(x='is_canceled', data=df, palette='coolwarm')
plt.title('Bookings vs Cancellations')
plt.xlabel('Is Canceled')
plt.ylabel('Count')
plt.grid(True)
plt.show()

# Create a new column 'reservation_month' to capture the month
df['reservation_month'] = df['reservation_status_date'].dt.to_period('M')

# Group by reservation month and count the number of bookings
monthly_bookings = df.groupby('reservation_month').size()

plt.figure(figsize=(10, 6))
monthly_bookings.plot(kind='line', color='blue', linewidth=2)
plt.title('Hotel Bookings Trend Over Time')
plt.xlabel('Month')
plt.ylabel('Number of Bookings')
plt.grid(True)
plt.xticks(rotation=45)
plt.show()


# Group by 'arrival_date_month' to get the number of bookings per month
monthly_booking_counts = df.groupby('arrival_date_month')['is_canceled'].count().sort_index()

# Group by month and calculate cancellations
monthly_cancellations = df.groupby('arrival_date_month')['is_canceled'].sum().sort_index()

# Create subplots to visualize both booking and cancellation trends
fig, ax = plt.subplots(2, 1, figsize=(10, 12))

# Plot monthly booking counts
monthly_booking_counts.plot(kind='bar', ax=ax[0], color='skyblue', edgecolor='black')
ax[0].set_title('Monthly Booking Counts')
ax[0].set_xlabel('Month')
ax[0].set_ylabel('Number of Bookings')
ax[0].grid(True)

# Plot monthly cancellations
monthly_cancellations.plot(kind='bar', ax=ax[1], color='salmon', edgecolor='black')
ax[1].set_title('Monthly Cancellations')
ax[1].set_xlabel('Month')
ax[1].set_ylabel('Number of Cancellations')
ax[1].grid(True)

plt.tight_layout()
plt.show()

# Calculate no-shows (bookings that were not canceled)
df['is_no_show'] = df['is_canceled'].apply(lambda x: 0 if x == 1 else 1)

# Count the number of no-shows
no_show_count = df['is_no_show'].sum()
total_count = df.shape[0]
no_show_percentage = (no_show_count / total_count) * 100

print(f"Total number of no-shows: {no_show_count}")
print(f"No-show percentage: {no_show_percentage:.2f}%")

# No-show trend analysis (over time)
monthly_no_shows = df.groupby('reservation_month')['is_no_show'].sum()

# Plot no-show trend over time
plt.figure(figsize=(10, 6))
monthly_no_shows.plot(kind='line', color='red', linewidth=2)
plt.title('No-Show Trend Over Time')
plt.xlabel('Month')
plt.ylabel('Number of No-Shows')
plt.grid(True)
plt.xticks(rotation=45)
plt.show()

# Create a new column 'total_stay_nights' to calculate total stay duration
df['total_stay_nights'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']

# Plot the distribution of total stay duration for canceled and non-canceled bookings
plt.figure(figsize=(10, 6))
sns.boxplot(x='is_canceled', y='total_stay_nights', data=df, palette='coolwarm')
plt.title('Stay Duration for Canceled vs. Non-Canceled Bookings')
plt.xlabel('Is Canceled')
plt.ylabel('Total Stay Duration (Nights)')
plt.grid(True)
plt.show()

