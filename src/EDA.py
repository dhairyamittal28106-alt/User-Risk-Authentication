import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# Load Dataset
df = pd.read_csv("src/auth_authz_failures_dataset.csv")


# 1. Login Success vs Failure
print(df["success"].value_counts())

plt.figure(figsize=(6,4))
sns.countplot(x="success", data=df)
plt.title("Login Success vs Failure")
plt.tight_layout()
plt.show()



# 2. Threat Level Distribution
print(df["threat_level"].value_counts())

plt.figure(figsize=(7,4))
sns.countplot(
    x="threat_level",
    data=df,
    order=df["threat_level"].value_counts().index
)

plt.title("Threat Level Distribution")
plt.tight_layout()
plt.show()



# 3. Suspicious Activity Analysis

plt.figure(figsize=(6,4))
sns.countplot(
    x="suspicious_activity",
    data=df
)

plt.title("Suspicious Activity Analysis")
plt.tight_layout()
plt.show()



# 4. Device Type Distribution

plt.figure(figsize=(7,4))

sns.countplot(
    x="device_type",
    data=df
)

plt.title("Device Type Distribution")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



# 5. Failed Login Attempts Distribution

plt.figure(figsize=(7,4))

sns.histplot(
    df["failed_attempts"],
    bins=20
)

plt.title("Failed Login Attempts Distribution")
plt.tight_layout()
plt.show()



# 6. MFA Enabled vs Threat Level

plt.figure(figsize=(7,4))

sns.countplot(
    x="mfa_enabled",
    hue="threat_level",
    data=df
)

plt.title("MFA Impact on Threat Level")
plt.tight_layout()
plt.show()





# 7 Feature Correlation Heatmap

plt.figure(figsize=(12,7))

sns.heatmap(
    df.corr(numeric_only=True),
    annot=True,
    cmap="coolwarm"
)

plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()