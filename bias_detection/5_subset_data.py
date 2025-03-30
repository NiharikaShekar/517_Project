import pandas as pd

# Load the original dataset
applicants = pd.read_csv('data/stackoverflow_full.csv')

# Select the first 200 rows
subset_data = applicants.head(200)

# Save the subset to a new CSV file
subset_data.to_csv('data/subsetdata.csv', index=False)

print("Subset data saved to 'subsetdata.csv'")
