
rule preprocess_metabolomics:
    input:
        "data/simulated_metabolomics/metabolomics.csv"
    output:
        "data/processed/metabolomics_preprocessed.csv"
    conda:
        "../../envs/simu.yaml"
    script:
        "../../scripts/preprocess_metabolomics.py"

