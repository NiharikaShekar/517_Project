import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the datasets (With correct column names)
fairness_metrics = pd.read_csv("predictions_output/fairness_metrics_results.csv")
fairness_metrics_counter = pd.read_csv("predictions_output/fairness_metrics_results_counterfactual.csv")
fairness_metrics_cot = pd.read_csv("predictions_output/fairness_metrics_cot.csv")

# Define labels for each dataset
labels_before = ["Man", "Nonbinary", "Woman"]  # For the first dataset (Before Bias Mitigation)
labels_counter = ["Man", "Woman"]  # For the second dataset (After Counterfactual Augmentation)
labels_cot = ["Man", "NonBinary", "Woman"]  # For the third dataset (After CoT Prompting)

# Function to plot graphs
# Function to plot graphs
def plot_fairness_metrics(df, title_prefix, save_prefix, labels):
    """
    Generates and saves fairness metric graphs for a given dataset.
    """
    plt.figure(figsize=(12, 4))

    # Demographic Parity
    plt.subplot(1, 3, 1)
    sns.barplot(x=labels, y=df["Demographic Parity"], palette="coolwarm")
    plt.title(f"{title_prefix} - Demographic Parity")
    plt.ylabel("Score")
    plt.xlabel("Gender")

    # Equalized Odds (False Positive Rate & False Negative Rate)
    plt.subplot(1, 3, 2)
    sns.barplot(x=labels, y=df["False Positive Rate (Equalized Odds)"], label="FPR", color="lightcoral")
    sns.barplot(x=labels, y=df["False Negative Rate (Equalized Odds)"], label="FNR", color="steelblue", alpha=0.7)
    plt.title(f"{title_prefix} - Equalized Odds")
    plt.ylabel("Score")
    plt.xlabel("Gender")
    plt.legend()

    # Predictive Rate Parity (Qualified & Unqualified Rates)
    plt.subplot(1, 3, 3)
    sns.barplot(x=labels, y=df["Qualified Rate (Predictive Rate Parity)"], label="Qualified", color="seagreen")
    sns.barplot(x=labels, y=df["Unqualified Rate (Predictive Rate Parity)"], label="Unqualified", color="goldenrod", alpha=0.7)
    plt.title(f"{title_prefix} - Predictive Rate Parity")
    plt.ylabel("Score")
    plt.xlabel("Gender")
    plt.legend()

    # Adjust title for the second graph to avoid overlap
    if save_prefix == "counter":
        plt.subplot(1, 3, 1)  # Title for the first graph (Demographic Parity)
        plt.title(f"{title_prefix}\n(Note: Male swapped with Woman)", fontsize=10)

    plt.tight_layout()
    plt.savefig(f"graphs/{save_prefix}_fairness_metrics.png")
    plt.show()

# Generate graphs for all three datasets
plot_fairness_metrics(fairness_metrics, "Before Bias Mitigation", "before", labels_before)
plot_fairness_metrics(fairness_metrics_counter, "After Counterfactual Augmentation", "counter", labels_counter)
plot_fairness_metrics(fairness_metrics_cot, "After CoT Prompting", "cot", labels_cot)

print("Graphs generated and saved in 'graphs/' folder.")
