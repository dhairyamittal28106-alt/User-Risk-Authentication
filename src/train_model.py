import pandas as pd
import joblib
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import RandomizedSearchCV
import shap

# --- 1. LOAD & ENCODE DATA ---
df = pd.read_csv("src/auth_authz_failures_dataset_cleaned.csv")

encoders = {}
for col in ["username", "ip_address", "location", "device_type", "browser"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le
joblib.dump(encoders, "src/label_encoders.pkl")

target_le = LabelEncoder()
df["threat_level"] = target_le.fit_transform(df["threat_level"])
joblib.dump(target_le, "src/target_encoder.pkl")

X = df.drop("threat_level", axis=1)
y = df["threat_level"]

# --- 2. MODEL TUNING & OPTIMIZATION ---
print("Starting Model Tuning & Optimization...")

# Define the hyperparameters to test
param_grid = {
    'max_depth': [3, 5, 7, 9],
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [50, 100, 200],
    'subsample': [0.8, 1.0]
}

base_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)

# RandomizedSearchCV tries different combinations to find the optimal model
tuned_model = RandomizedSearchCV(
    estimator=base_model, 
    param_distributions=param_grid, 
    n_iter=10, # Number of parameter settings that are sampled
    scoring='accuracy', 
    cv=3,      # 3-fold cross-validation
    verbose=1, 
    random_state=42
)

# Train the tuned model
tuned_model.fit(X, y)
best_xgb = tuned_model.best_estimator_

print(f"\nBest Parameters Found: {tuned_model.best_params_}")

# Save the optimized model for app.py to use
joblib.dump(best_xgb, "src/authentication_model.pkl")
print("Optimized model saved successfully!")

# --- 3. SHAP & FEATURE IMPORTANCE ---
print("\nCalculating Feature Importance & SHAP values...")

# Print basic Feature Importance to terminal
importance = best_xgb.feature_importances_
feat_imp_df = pd.DataFrame({'Feature': X.columns, 'Importance': importance}).sort_values(by='Importance', ascending=False)
print("\n--- Basic Feature Importance ---")
print(feat_imp_df)

# Generate SHAP values
explainer = shap.TreeExplainer(best_xgb)
shap_values = explainer.shap_values(X)

# FIX: Check if SHAP returned a 3D array (happens in multi-class XGBoost)
# and convert it into a list of 2D arrays so summary_plot can read it.
if len(shap_values.shape) == 3:
    shap_values = [shap_values[:, :, i] for i in range(shap_values.shape[2])]

# Create and save a SHAP Summary Plot
plt.figure(figsize=(10, 6))
# plot_type="bar" is required when passing multi-class lists to summary_plot
shap.summary_plot(shap_values, X, show=False, plot_type="bar")
plt.tight_layout()
plt.savefig("src/shap_summary.png")
print("\nSHAP summary plot saved as 'src/shap_summary.png'!")