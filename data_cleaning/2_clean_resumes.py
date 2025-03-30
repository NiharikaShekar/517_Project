import pandas as pd
import re

# Load resumes
resumes = pd.read_csv('data/UpdatedResumeDataSet.csv')

# Clean resume text
def clean_text(text):
    text = re.sub(r'\n', ' ', str(text))  # Remove line breaks
    text = re.sub(r'\W', ' ', text)       # Remove special characters
    text = re.sub(r'\s+', ' ', text)      # Remove multiple spaces
    return text.lower().strip()

# Apply cleaning
resumes['Resume'] = resumes['Resume'].apply(clean_text)

# Save cleaned resumes
resumes[['Resume']].to_csv('data/cleaned_resumes.csv', index=False)
print("Resumes cleaned and saved to 'cleaned_resumes.csv'")
