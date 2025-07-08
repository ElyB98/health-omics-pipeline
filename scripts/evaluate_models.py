# Import necessary libraries
import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.metrics import roc_curve, auc, ConfusionMatrixDisplay
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Load the preprocessed integrated dataset
df = pd.read_csv("data/processed/integrated_dataset.csv", index_col=0)

# Convert all columns (except "sex" and "outcome") to float
exclude = {"sex", "outcome"}
for col in df.columns:
    if col not in exclude:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Select only numeric columns
X = df.select_dtypes(include="number")
if "outcome" in X.columns:
    X = X.drop(columns=["outcome"])

# Target
y = df["outcome"]


# Drop columns with all missing values
X = X.dropna(axis=1, how="all")

# Then impute remaining missing values
imputer = SimpleImputer(strategy="mean")
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)


# Define function to evaluate a model using cross-validated F1 score
def evaluate_model(model, X, y, name):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  # Ensure class balance in folds
    scores = cross_val_score(model, X, y, cv=skf, scoring="f1")
    print(f"{name} - Mean F1: {scores.mean():.2f} (+/- {scores.std():.2f})")
    return scores.mean()

# Evaluate baseline classifiers
scores = {
    "LogReg": evaluate_model(LogisticRegression(max_iter=500), X, y, "LogisticRegression"),
    "RF": evaluate_model(RandomForestClassifier(), X, y, "RandomForest"),
    "SVM": evaluate_model(SVC(), X, y, "SVM"),
    # "XGB": evaluate_model(XGBClassifier(eval_metric="logloss"), X, y, "XGBoost"),  # Optional
}

# Hyperparameter optimization for Random Forest
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [5, 10, None]
}
grid = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, scoring="f1")
grid.fit(X, y)
print("Best RF params:", grid.best_params_)

# Save the best Random Forest model
joblib.dump(grid.best_estimator_, "models/best_randomforest.pkl")

# Train logistic regression to plot the ROC curve
model = LogisticRegression(max_iter=500).fit(X, y)
y_score = model.predict_proba(X)[:, 1]  # Get probability estimates
fpr, tpr, _ = roc_curve(y, y_score)
roc_auc = auc(fpr, tpr)

# Save ROC curve plot
os.makedirs("results", exist_ok=True)
plt.figure()
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Logistic Regression')
plt.legend()
plt.savefig("results/roc_logreg.png")

# Plot and save confusion matrix
y_pred = model.predict(X)
ConfusionMatrixDisplay.from_predictions(y, y_pred)
plt.savefig("results/confusion_matrix_logreg.png")

# Save comparison of model F1 scores
results_df = pd.DataFrame({
    "model": list(scores.keys()),
    "mean_f1": list(scores.values())
})
results_df.to_csv("results/model_comparison.csv", index=False)
print("✅ Results saved successfully.")
