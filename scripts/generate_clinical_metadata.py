# scripts/generate_clinical_metadata.py
import pandas as pd
import numpy as np
from faker import Faker

fake = Faker()
np.random.seed(42)

samples = [f"sample_{i}" for i in range(1, 101)]
ages = np.random.randint(30, 80, size=100) 
sexes = np.random.choice(["M", "F"], size=100)
outcomes = np.random.choice(["responder", "non-responder"], size=100)

df = pd.DataFrame({
    "sample_id": samples,
    "age": ages,
    "sex": sexes,
    "outcome": outcomes
})

df.to_csv("data/clinical_metadata.csv", index=False)
