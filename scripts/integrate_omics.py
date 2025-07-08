import pandas as pd
import os
import argparse

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--rnaseq", required=True)
parser.add_argument("--clinical", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()

# Load data
rna = pd.read_csv(args.rnaseq, index_col=0)
clinical_df = pd.read_csv(args.clinical)

# Drop duplicates on sample_id
clinical_df = clinical_df.drop_duplicates(subset="sample_id", keep="first")
print(f"ℹ️ Clinical metadata cleaned: {clinical_df.shape[0]} unique samples")

# Set index for merging
clinical_df.set_index("sample_id", inplace=True)

# Truncate and clean RNA index to match format TCGA-XX-YYYY
rna.index = rna.index.to_series().str.extract(r'^(TCGA-\w\w-\w\w\w\w)')[0].str.strip()
rna = rna[~rna.index.isna()]  # remove NaNs if any

# Remove duplicates
rna = rna[~rna.index.duplicated(keep='first')]

# Check intersection
common_ids = set(rna.index) & set(clinical_df.index)
print(f"🔗 Found {len(common_ids)} common sample IDs")
print("🧬 RNA IDs (first 5):", rna.index.unique()[:5].tolist())
print("🏥 Clinical IDs (first 5):", clinical_df.index.unique()[:5].tolist())

# Filter both datasets to common samples
rna_common = rna.loc[rna.index.isin(common_ids)].copy()
clinical_common = clinical_df.loc[rna_common.index].copy()

# Specify the actual target column
target_col = "outcome"
assert target_col in clinical_common.columns, f"❌ Missing target column: {target_col}"

# Separate features and target
features = clinical_common.drop(columns=[target_col])
target = clinical_common[target_col]

# Encode categorical variables (excluding the target)
categorical_cols = features.select_dtypes(include="object").columns
features = pd.get_dummies(features, columns=categorical_cols, drop_first=True)

# Impute missing numeric values
features = features.fillna(features.median(numeric_only=True))

# Recombine features and target
clinical_common = features
clinical_common[target_col] = target


# Sanity check
print("🧬 RNA shape:", rna_common.shape)
print("🏥 Clinical shape:", clinical_common.shape)
assert rna_common.index.is_unique, "RNA-seq index is not unique!"
assert clinical_common.index.is_unique, "Clinical metadata index is not unique!"

# Rename RNA columns
rna_common.columns = [f"{col}_rna" for col in rna_common.columns]

# Merge
merged = pd.concat([rna_common, clinical_common], axis=1)

# Save
os.makedirs(os.path.dirname(args.output), exist_ok=True)
merged.to_csv(args.output)
print(f"✅ Integration saved to: {args.output}")
