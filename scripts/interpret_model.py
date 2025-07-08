# scripts/interpret_model.py

import pandas as pd
import shap
import matplotlib.pyplot as plt
import joblib
import os
import sys

# Argument parsing from Snakemake 
model_path = sys.argv[1] if len(sys.argv) > 1 else snakemake.input.model
data_path = sys.argv[2] if len(sys.argv) > 2 else snakemake.input.data
gini_output = snakemake.output[0]
shap_output = snakemake.output[1]

# Load model and data 
model = joblib.load(model_path)
data = pd.read_csv(data_path, index_col=0)

# Drop target column 
if "outcome" not in data.columns:
    raise ValueError("'outcome' column not found in input data.")

y = data["outcome"]
X = data.drop(columns=["outcome"])


# Gini importance 
importances = model.feature_importances_
indices = importances.argsort()[::-1][:20]  # top 20

plt.figure(figsize=(10, 6))
plt.title("Top 20 Gini Feature Importances")
plt.bar(range(len(indices)), importances[indices])
plt.xticks(range(len(indices)), [X.columns[i] for i in indices], rotation=90)
plt.tight_layout()
plt.savefig(gini_output)
plt.close()

# SHAP values 
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)


# Handle binary or single output
if isinstance(shap_values, list):
    shap_values_to_plot = shap_values[1]  
else:
    shap_values_to_plot = shap_values

shap.summary_plot(shap_values_to_plot, X, show=False)

# Summary plot for class 1 (assumes binary classification)
plt.tight_layout()
plt.savefig(shap_output, bbox_inches="tight")
plt.close()
