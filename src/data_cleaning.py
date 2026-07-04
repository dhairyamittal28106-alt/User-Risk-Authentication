import pandas as pd
import numpy as np

df = pd.read_csv("src/auth_authz_failures_dataset.csv")

columns_to_keep = [
    "username", 
    "ip_address", 
    "location", 
    "failed_attempts", 
    "device_type", 
    "browser", 
    "mfa_enabled", 
    "threat_level"
]

df = df[columns_to_keep]

df["username"] = df["username"].fillna("Unknown")
df["ip_address"] = df["ip_address"].fillna("0.0.0.0")
df["location"] = df["location"].fillna("Unknown")
df["failed_attempts"] = df["failed_attempts"].fillna(0)
df["device_type"] = df["device_type"].fillna("Unknown")
df["browser"] = df["browser"].fillna("Unknown")
df["mfa_enabled"] = df["mfa_enabled"].fillna(0)
df["threat_level"] = df["threat_level"].fillna("Low")

df = df.drop_duplicates()

df.to_csv("src/auth_authz_failures_dataset_cleaned.csv", index=False)

print("Cleaned file saved! Shape:", df.shape)