import pandas as pd
from ucimlrepo import fetch_ucirepo

# Fetch dataset 
default_of_credit_card_clients = fetch_ucirepo(id=350)

# Data (as pandas dataframes)
X = default_of_credit_card_clients.data.features
y = default_of_credit_card_clients.data.targets
print("\n--- Content of y (target) DataFrame ---")
print(y)

# Concatenate X and y for easier initial exploration
print("\n--- Columns in y (target) DataFrame ---")
print(y.columns)

df = pd.concat([X, y], axis=1)

# Display basic information
print("\n--- Dataset Head ---")
print(df.head())

print("\n--- All Columns in DataFrame (df) ---")
print(df.columns)

print("\n--- Dataset Info ---")
df.info()

print("\n--- Missing Values ---")
print(df.isnull().sum())

print("\n--- Descriptive Statistics ---")
print(df.describe())

print("\n--- Target Variable Distribution ---")
# Assuming the correct column name is 'default payment next month' based on the dataset description
# If the above print statement shows a different name, update this line accordingly.
print(df['Y'].value_counts())
print(df['Y'].value_counts(normalize=True) * 100)

# Save the raw data for future use (optional)
df.to_csv('loan_default_data.csv', index=False)
print("\nRaw data saved to loan_default_data.csv")