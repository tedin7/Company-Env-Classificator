# Import necessary libraries
import pandas as pd

# Read the dataset into a pandas DataFrame
df = pd.read_csv('/home/tomd/Documenti/GitHub/Company-Env-Classificator/Data/filtered_dataset.csv')
print(df.columns)
# Get the total number of rows
total_rows = len(df)

# Create a dictionary to store the number of unique values for each column
unique_value_counts = {}

# Iterate over columns and count unique values
for column in df.columns:
    unique_value_counts[column] = df[column].nunique()

# Sort the dictionary by the number of unique values in descending order
sorted_counts = sorted(unique_value_counts.items(), key=lambda x: x[1], reverse=True)

# Display the total number of rows
print(f"Total number of rows: {total_rows}\n")

# Display column information ordered by the number of unique values
for column, count in sorted_counts:
    # Count missing values in the column
    missing_values = df[column].isnull().sum()
    
    # Convert the column to a string type
    df[column] = df[column].astype(str)
    
    # Get unique values in the column
    unique_values = df[column].unique()
    
    # Display column name, number of missing values, and unique values
    print(f"Column: {column}")
    print(f"Number of missing values: {missing_values}")
    print(f"Number of unique values: {count}")
    print(f"Unique values: {unique_values}\n")
