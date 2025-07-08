import pandas as pd
import numpy as np
import os

try:
    input_path = snakemake.input[0]
    output_path = snakemake.output[0]

    # Load input file
    df = pd.read_csv(input_path, index_col=0)
    df.index.name = "sample_id"
    df.index = df.index.astype(str)

    # Log2 transformation
    df_log = np.log2(df + 1)

    # Z-score normalization
    df_scaled = (df_log - df_log.mean()) / df_log.std()

    # Save the result
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_scaled.to_csv(output_path)

    print(f"Preprocessing completed: {output_path}")

except Exception as e:
    print(f"Error in preprocess_metabolomics.py: {e}")
    raise
