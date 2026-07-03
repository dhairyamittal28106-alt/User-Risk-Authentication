import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


df=pd.read_csv("src/auth_authz_failures_dataset.csv")
print(df.head(3))
df.info()
print(df.isnull().sum())
print("Duplicate rows:", df.duplicated().sum())
df=df.drop("error_code", axis=1)
df.head()
print("Infinite values:", np.isinf(df.select_dtypes(include=[np.number])).sum().sum())
df.to_csv("src/auth_authz_failures_dataset_cleaned.csv", index=False)
print(df.columns)