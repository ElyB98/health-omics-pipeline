import pandas as pd
import os

# Convert Clinical Metadata 

clinical_input = "data/raw/TCGA-BRCA.clinical.tsv"
clinical_output = "data/processed/clinical_metadata.csv"

# Load and clean clinical metadata
clinical_df = pd.read_csv(clinical_input, sep="\t", low_memory=False)
clinical_df = clinical_df[[
    "submitter_id",
    "gender.demographic",
    "vital_status.demographic"
]].copy()


# Rename columns
clinical_df.columns = ["sample_id", "sex", "vital_status"]

# Clean sample IDs (no truncation, just strip)
clinical_df["sample_id"] = clinical_df["sample_id"].str.strip()

# Drop missing values
clinical_df = clinical_df.dropna(subset=["sample_id", "sex", "vital_status"])

# Map survival outcome
clinical_df["outcome"] = clinical_df["vital_status"].str.lower().map({
    "alive": 1,
    "dead": 0
})

# Drop if outcome couldn't be mapped
clinical_df = clinical_df.dropna(subset=["outcome"])

# Keep relevant columns
clinical_df = clinical_df[["sample_id", "outcome", "sex"]]


# Save
os.makedirs(os.path.dirname(clinical_output), exist_ok=True)
clinical_df.to_csv(clinical_output, index=False)
print(f"✅ Clinical metadata saved to: {clinical_output}")


# Normalize RNA-seq Data 

rnaseq_input = "data/raw/TCGA-BRCA.star_fpkm-uq.tsv"
rnaseq_output = "data/processed/rnaseq_normalized.csv"

# Load RNA-seq and transpose
rnaseq_df = pd.read_csv(rnaseq_input, sep="\t", index_col=0).transpose()
rnaseq_df.index.name = "sample_id"
rnaseq_df.index = rnaseq_df.index.astype(str)

# Convert to numeric and normalize
rnaseq_df = rnaseq_df.apply(pd.to_numeric, errors="coerce")
rnaseq_scaled = (rnaseq_df - rnaseq_df.mean()) / rnaseq_df.std()

# Save
os.makedirs(os.path.dirname(rnaseq_output), exist_ok=True)
rnaseq_scaled.to_csv(rnaseq_output)
print(f"✅ RNA-seq normalized and saved to: {rnaseq_output}")
