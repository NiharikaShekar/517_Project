import pandas as pd

# Load the Ollama predictions (contains Gender and Decision)
pred_df = pd.read_csv('predictions_output/ollama_predictions_counterfactual.csv')

# Load the original dataset (contains Gender and Employed columns)
original_data = pd.read_csv('data/stackoverflow_full.csv')  # Or whichever original file you used

# Ensure the 'Employed' column is available from the original dataset
# Merge the original dataset's 'Employed' column with the predictions dataframe based on the index
merged_df = pred_df.copy()
merged_df['Employed'] = original_data['Employed'][:len(pred_df)]  # Assuming first 200 rows match

# Now you can calculate Equalized Odds, Demographic Parity, or any other fairness metric using the merged dataframe.

# Function to calculate Demographic Parity
def demographic_parity(pred_df):
    dp = pred_df.groupby('CounterfactualGender')['Decision'].mean()  # Mean proportion of "Yes" decisions across all genders
    return dp

# Function to calculate Equalized Odds (False Positive Rate & False Negative Rate)
def equalized_odds(pred_df):
    fp = pred_df[(pred_df['Decision'] == 1) & (pred_df['Employed'] == 0)]  # False positives (unqualified hired)
    fn = pred_df[(pred_df['Decision'] == 0) & (pred_df['Employed'] == 1)]  # False negatives (qualified rejected)
    
    # Calculate rates per gender
    fp_rate = fp.groupby('CounterfactualGender').size() / pred_df.groupby('CounterfactualGender').size()
    fn_rate = fn.groupby('CounterfactualGender').size() / pred_df.groupby('CounterfactualGender').size()
    
    return fp_rate, fn_rate

# Function to calculate Predictive Rate Parity
def predictive_rate_parity(pred_df):
    qualified = pred_df[pred_df['Employed'] == 1]
    unqualified = pred_df[pred_df['Employed'] == 0]
    
    qualified_rate = qualified.groupby('CounterfactualGender')['Decision'].mean()  # Rate of positive predictions for qualified candidates
    unqualified_rate = unqualified.groupby('CounterfactualGender')['Decision'].mean()  # Rate of positive predictions for unqualified candidates
    
    return qualified_rate, unqualified_rate

# Calculate and save all fairness metrics
print("Calculating fairness metrics...\n")

# Demographic Parity Calculation
dp = demographic_parity(merged_df)

# Equalized Odds Calculation
fp_rate, fn_rate = equalized_odds(merged_df)

# Predictive Rate Parity Calculation
qualified_rate, unqualified_rate = predictive_rate_parity(merged_df)

# Prepare results to save to CSV
results = {
    'Demographic Parity': dp,
    'False Positive Rate (Equalized Odds)': fp_rate,
    'False Negative Rate (Equalized Odds)': fn_rate,
    'Qualified Rate (Predictive Rate Parity)': qualified_rate,
    'Unqualified Rate (Predictive Rate Parity)': unqualified_rate
}

# Convert the results to DataFrame
results_df = pd.DataFrame(results)

# Save the results to CSV
results_df.to_csv('predictions_output/fairness_metrics_results_counterfactual.csv', index=False)

print(f"Fairness metrics saved to 'predictions_output/fairness_metrics_results_counterfactual.csv'")
