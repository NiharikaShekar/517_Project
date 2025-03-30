import pandas as pd

# Load the subset data
applicants = pd.read_csv('data/subsetdata.csv')

# Clean up any leading/trailing spaces and capitalize gender values consistently
applicants['Gender'] = applicants['Gender'].str.strip().str.capitalize()

# Check if the gender values are as expected after cleaning
print("Unique Gender values before modification:", applicants['Gender'].unique())

# Create counterfactuals: Swap gender values
def counterfactual_gender(row):
    if row['Gender'] == 'Man':
        return 'Woman'
    elif row['Gender'] == 'Woman':
        return 'Man'
    elif row['Gender'] == 'Nonbinary':  # Be sure it's consistent with how NonBinary is stored
        return 'Man'  # For simplicity, we'll switch NonBinary to Man here
    return row['Gender']  # Return unchanged if gender is neither of the above

# Apply counterfactual gender modification
applicants['CounterfactualGender'] = applicants.apply(counterfactual_gender, axis=1)

# Check the unique values in CounterfactualGender to ensure the transformation worked
print("Unique values in CounterfactualGender after modification:", applicants['CounterfactualGender'].unique())

# Save the augmented data (counterfactuals)
applicants.to_csv('data/counterfactual_subset.csv', index=False)
print("Counterfactual data (gender modification) saved to 'counterfactual_subset.csv'")
