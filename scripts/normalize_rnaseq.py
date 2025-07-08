import pandas as pd
import os

try:
    output_path = snakemake.output[0]
    input_path = "data/raw/TCGA-BRCA.star_fpkm-uq.tsv"

    # Load raw RNA-seq data
    df = pd.read_csv(input_path, sep="\t", index_col=0)

    # Transpose to have samples as rows
    df = df.transpose()
    df.index.name = "sample_id"

    # Clean sample IDs to extract TCGA-XX-YYYY format
    df.index = df.index.to_series().str.extract(r'^(TCGA-\w\w-\w\w\w\w)')[0].str.strip()

    # Drop any rows with NaN as index (i.e., unparseable sample IDs)
    df = df[~df.index.isna()]
    df = df[~df.index.duplicated(keep="first")]

    # Convert to numeric and normalize (z-score)
    df = df.apply(pd.to_numeric, errors="coerce")
    df = (df - df.mean()) / df.std()

    # Save normalized file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path)

    print(f"RNA-seq normalized and saved to: {output_path}")

except Exception as e:
    print(f"Error in normalize_rnaseq.py: {e}")
    raise
