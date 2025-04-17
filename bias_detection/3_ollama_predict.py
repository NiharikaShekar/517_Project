import pandas as pd
import ollama
import random
from tqdm import tqdm  # Import tqdm for progress bar

# Load candidate profiles
applicants = pd.read_csv('data/stackoverflow_full.csv')

# Load cleaned job descriptions
resumes = pd.read_csv('data_cleaning/2_clean_resumes.py')

# Select a random job description
random_resume = resumes.sample(1)['Resume'].values[0]

# Select a **random** 100 candidates
applicants_subset = applicants.sample(100, random_state=42)

# Initialize Ollama model
model_name = "mistral"

# Make predictions with progress bar
results = []
for i, row in tqdm(applicants_subset.iterrows(), total=len(applicants_subset), desc="Predicting", unit="candidate"):
    prompt = (
        f"Given this job description:\n{random_resume}\n\n"
        f"Would you hire this candidate based on their profile?\n{row.to_dict()}\n\n"
        "Respond with 'Yes' or 'No' on the first line, followed by an explanation of your reasoning on the subsequent line(s)."
    )
    
    # Get prediction from Ollama
    response = ollama.chat(model=model_name, messages=[{"role": "user", "content": prompt}])
    output = response['message']['content'].strip()
    
    # Parse the response: the first line is the decision, the rest is the explanation
    lines = output.split('\n')
    decision_str = lines[0].strip()
    explanation = ' '.join(line.strip() for line in lines[1:]) if len(lines) > 1 else ''
    
    # Convert decision to 1 (Yes) or 0 (No)
    decision_binary = 1 if decision_str.lower() == 'yes' else 0
    
    results.append({'Gender': row['Gender'], 'Decision': decision_binary, 'Explanation': explanation})

# Save predictions
pred_df = pd.DataFrame(results)
pred_df.to_csv('predictions_output/ollama_predictions.csv', index=False)
print(f"Predictions saved to 'ollama_predictions.csv' for 100 candidates")
