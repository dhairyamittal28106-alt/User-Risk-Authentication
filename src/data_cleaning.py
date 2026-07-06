import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("src/auth_authz_failures_dataset.csv")

print(df.shape)
df.info()
print("Rows:",df.shape[0])

print("Columns:",df.shape[1])

print(df.describe())
df.info()

print("Duplicates:",df.duplicated().sum())

print("Missing Values:",df.isnull().sum())
print(df.dtypes)
df["failure_reason"] = df["failure_reason"].fillna("Unknown")
df["error_code"] = df["error_code"].fillna("Unknown")

df = df.drop(columns=["timestamp"])
df.to_csv("src/cleaned_dataset.csv", index=False)