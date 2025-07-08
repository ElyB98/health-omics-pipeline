# scripts/simulate_metabolomics.py
import os
import numpy as np
import pandas as pd

os.makedirs("data/simulated_metabolomics", exist_ok=True)

np.random.seed(42)
samples = [f"sample_{i}" for i in range(1, 101)] 
features = [f"metabolite_{j}" for j in range(1, 101)]

data = np.random.normal(loc=0, scale=1, size=(100, 100))
df = pd.DataFrame(data, index=samples, columns=features)

df.to_csv("data/simulated_metabolomics/metabolomics.csv")
