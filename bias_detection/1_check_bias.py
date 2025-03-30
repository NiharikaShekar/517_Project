import pandas as pd

# Load dataset
df = pd.read_csv('data/stackoverflow_full.csv')

# Check gender balance
gender_counts = df['Gender'].value_counts()
print("Gender Distribution:\n", gender_counts)

# Check hiring bias
hiring_bias = df.groupby('Gender')['Employed'].value_counts(normalize=True).unstack()
print("\nHiring Bias:\n", hiring_bias)

# Save results
gender_counts.to_csv('data/gender_distribution.csv')
hiring_bias.to_csv('data/hiring_bias.csv')
