import pandas as pd
# Load new CoT-based predictions
pred_df = pd.read_csv('predictions_output/ollama_predictions_cot.csv')

# Function to calculate Demographic Parity
def demographic_parity(pred_df):
    dp = pred_df.groupby('Gender')['Decision'].mean()
    return dp

# Function to calculate Equalized Odds
def equalized_odds(pred_df):
    fp = pred_df[(pred_df['Decision'] == 1) & (pred_df['Employed'] == 0)]
    fn = pred_df[(pred_df['Decision'] == 0) & (pred_df['Employed'] == 1)]
    
    fp_rate = fp.groupby('Gender').size() / pred_df.groupby('Gender').size()
    fn_rate = fn.groupby('Gender').size() / pred_df.groupby('Gender').size()
    
    return fp_rate.fillna(0), fn_rate.fillna(0)

# Function to calculate Predictive Rate Parity
def predictive_rate_parity(pred_df):
    qualified = pred_df[pred_df['Employed'] == 1]
    unqualified = pred_df[pred_df['Employed'] == 0]
    
    qualified_rate = qualified.groupby('Gender')['Decision'].mean()
    unqualified_rate = unqualified.groupby('Gender')['Decision'].mean()
    
    return qualified_rate.fillna(0), unqualified_rate.fillna(0)

# Calculate fairness metrics
dp = demographic_parity(pred_df)
fp_rate, fn_rate = equalized_odds(pred_df)
qualified_rate, unqualified_rate = predictive_rate_parity(pred_df)

# Save fairness results to CSV
fairness_results = pd.DataFrame({
    'Demographic Parity': dp,
    'False Positive Rate (Equalized Odds)': fp_rate,
    'False Negative Rate (Equalized Odds)': fn_rate,
    'Qualified Rate (Predictive Rate Parity)': qualified_rate,
    'Unqualified Rate (Predictive Rate Parity)': unqualified_rate
})

fairness_results.to_csv("predictions_output/fairness_metrics_cot.csv", index=True)

print("Fairness metrics after CoT saved successfully.")
