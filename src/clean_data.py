import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("src/auth_authz_failures_dataset.csv")

y = df["success"]
print(df["blocked"].value_counts(normalize=True))
