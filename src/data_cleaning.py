import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


df=pd.read_csv("src/auth_authz_failures_dataset.csv")
print(df.shape)
df.info()
print("Rows:",df.shape[0])

print("Columns:",df.shape[1])

print(df.isnull().sum())
df[df.isnull().any(axis=1)]
df["failure_reason"] = df["failure_reason"].fillna("None")
df["error_code"] = df["error_code"].fillna("None")

print("Duplicate rows:", df.duplicated().sum())
df=df.drop("error_code", axis=1)
df.head()
print("Infinite values:", np.isinf(df.select_dtypes(include=[np.number])).sum().sum())



df.to_csv("src/cleaned.csv", index=False)