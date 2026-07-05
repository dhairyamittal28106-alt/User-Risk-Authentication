import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("src/auth_authz_failures_dataset.csv")

print("Rows:", df.shape[0])