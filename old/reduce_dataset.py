import pandas as pd

# Load the original dataset
df = pd.read_csv('Data/translated_dataset_es.csv')

# Select the desired columns
selected_columns = ['id', 'about', 'keywords', 'Label']

# Create a new DataFrame with the selected columns
new_df = df[selected_columns].copy()
new_df.dropna(inplace=True)

# Filter out rows where the 'keywords' column contains only empty lists or dictionaries
new_df = new_df[new_df['keywords'].apply(lambda x: x not in ([], {}))]

# Save the filtered dataset to a new CSV file
new_df.to_csv('Data/filtered_dataset_es.csv', index=False)

# Display the first few rows of the new dataset
print(new_df.head())
