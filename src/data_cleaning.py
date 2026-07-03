import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


df=pd.read_csv("src/CICIDS2017.csv")
print(df.head(3))
df.info()
print(df.isnull().sum())
print("Duplicate rows:", df.duplicated().sum())

print("Infinite values:", np.isinf(df.select_dtypes(include=[np.number])).sum().sum())
df.to_csv("src/CICIDS2017_cleaned.csv", index=False)
