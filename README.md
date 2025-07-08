# Health-Omics Pipeline

This project provides a modular and reproducible Snakemake pipeline for preprocessing, integrating, exploring, and modeling multi-omics data (RNA-seq, metabolomics) combined with clinical metadata. It supports both **simulated datasets** (for prototyping) and **real-world biomedical datasets** (e.g., TCGA BRCA), making it suitable for research, freelance projects, and applied data science portfolios.

## Overview

### Objectives

- Simulate synthetic RNA-seq and metabolomics datasets to prototype the pipeline
- Preprocess and integrate **real TCGA RNA-seq** and **clinical metadata**
- Perform exploratory data analysis (EDA) and dimensionality reduction
- Train and evaluate machine learning models to predict clinical outcomes (e.g., survival)
- Interpret model predictions using feature attribution (Gini & SHAP)

## Project Structure

```text
health-omics-pipeline/
├── data/                  
│   ├── raw/                  # Real-world TCGA data
│   │   └── TCGA-BRCA.star_fpkm-uq.tsv
│   ├── processed/            # Normalized & merged datasets
│   │   ├── rnaseq_normalized.csv
│   │   └── integrated_dataset.csv
│   ├── clinical_metadata.csv
│   ├── simulated_rnaseq/     # Simulated RNA-seq FASTQ files
│   ├── simulated_metabolomics/
│   └── clinical_metadata.csv # for simulated data
├── envs/                   # Conda environments
├── models/                 # Trained models (.pkl)
├── notebooks/
│   ├── eda_simulated_dataset.ipynb
│   └── eda_integrated_dataset.ipynb
├── results/
│   ├── roc_logreg.png
│   ├── confusion_matrix_logreg.png
│   ├── pca_plot.png
│   ├── shap_summary_plot.png
│   ├── feature_importance_gini.png
│   └── eda_integrated_dataset.html
├── rules/                  # Snakemake rules (e.g., preprocessing.smk)
├── scripts/                # Python & R scripts
├── workflow/
│   └── Snakefile
└── README.md
```



## Pipeline Summary

### Phase 1: Data Simulation

- `simulate_rnaseq.R`: Generates 100 `.fq` files representing RNA-seq reads
- `simulate_metabolomics.py`: Simulates 100 samples × 100 metabolite levels
- `generate_clinical_metadata.py`: Produces a file with age, sex, outcome for 100 patients

### Phase 2: Preprocessing

- RNA-seq normalized to TPM-like values → `rnaseq_normalized.csv`
- Metabolomics standardized with Z-score → `metabolomics_preprocessed.csv`

### Phase 3: Multi-Omics Integration

- Merge RNA-seq, metabolomics, and clinical metadata
- Result: `integrated_dataset.csv` (100 samples × 203 features)

### Phase 4: Exploratory Analysis & Modeling

- Missing value imputation
- PCA (Principal Component Analysis) → `results/pca_plot.png`
- Binary classification (predict `outcome`) using:
  - Logistic Regression
  - Random Forest
- Saved models → `models/*.pkl`
- Performance metrics:
  - ROC curve → `results/roc_logreg.png`
  - Confusion matrix → `results/confusion_matrix_logreg.png`
  - Model comparison → `results/model_comparison.csv`
  
  
### Phase 5: Real Data Analysis — TCGA Breast Cancer (BRCA)

After experimenting with simulated omics data, we transitioned to real-world biomedical data to demonstrate the pipeline’s practical relevance.

#### Data Source
We used data from The Cancer Genome Atlas (TCGA), specifically the **TCGA-BRCA** (breast invasive carcinoma) cohort.

- **RNA-Seq Data**: Raw FPKM-UQ gene expression data from GDC (Genomic Data Commons)
- **Clinical Metadata**: Clinical status, sex, and survival information (via cBioPortal)
- **Target Variable**: `OS_status` (Overall Survival), a binary variable

#### 🔧 Integration & Preprocessing
- Cleaned and harmonized sample identifiers
- Normalized gene expression values
- One-hot encoded clinical variables
- Imputed missing values

Result: `integrated_dataset.csv` (final dataset used in modeling)

#### 📊 Exploratory Analysis
Notebook: `eda_integrated_dataset.ipynb`

- PCA visualization
- Sample distribution checks
- Feature density & correlations

#### 🤖 Modeling and Interpretation
We trained models to predict `OS_status` using omics and clinical features:

- Logistic Regression
- Random Forest (best F1 score)
- Support Vector Machine

We also interpreted feature importance using:

- **Gini Importance**
- **SHAP values** (for local/global explanations)

All outputs are located in the `results/` directory.

---

### Phase 6: Optional Improvements

The following improvements were originally designed for the simulated-data pipeline but remain applicable to real-world modeling for enhanced robustness:

- Add classifiers: XGBoost, LightGBM
- Apply k-fold cross-validation
- Use `GridSearchCV` / `RandomizedSearchCV` for hyperparameter tuning
- Add L1/L2 regularization or recursive feature elimination (RFE)
- Benchmark model stability with different seeds

---

## Reproducibility

This project uses [Snakemake](https://snakemake.readthedocs.io) and conda environments to ensure full reproducibility.

To run the pipeline end-to-end:

```bash
snakemake --use-conda --cores 4

