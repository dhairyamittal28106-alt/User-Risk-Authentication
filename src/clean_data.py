import pandas as pd

df = pd.read_csv("src/auth_authz_failures_dataset.csv")

cols = ["username", "ip_address", "location", "failed_attempts", "device_type", "browser", "mfa_enabled", "threat_level"]
df = df[cols].fillna({"username": "Unknown", "ip_address": "0.0.0.0", "location": "Unknown", "failed_attempts": 0, "device_type": "Unknown", "browser": "Unknown", "mfa_enabled": 0, "threat_level": "Low"})

df.to_csv("src/auth_authz_failures_dataset_cleaned.csv", index=False)