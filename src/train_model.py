import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score   
)
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import RandomizedSearchCV
import shap

df = pd.read_csv("src/cleaned_dataset.csv")

encoders = {}
for col in ["username", "ip_address", "location", "device_type", "browser"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le
joblib.dump(encoders, "src/label_encoders.pkl")

target_le = LabelEncoder()
df["threat_level"] = target_le.fit_transform(df["threat_level"])
joblib.dump(target_le, "src/target_encoder.pkl")

features = [
    "username",
    "ip_address",
    "location",
    "failed_attempts",
    "device_type",
    "browser",
    "mfa_enabled"
]

X = df[features]
y = df["threat_level"]
#train-test split
X_train,X_test,y_train,y_test=train_test_split(
    X,
    y,
    random_state=42,
    test_size=0.2,
    stratify=y
)

#Starting model training
print(".........Logistic Regression.........")
lr_model=LogisticRegression(max_iter=1000)
lr_model.fit(X_train,y_train)
y_pred_lr=lr_model.predict(X_test)
lr_pred = lr_model.predict(X_test)

lr_acc = accuracy_score(y_test, lr_pred)
lr_pre = precision_score(y_test, lr_pred, average='weighted')
lr_rec = recall_score(y_test, lr_pred, average='weighted')
lr_f1 = f1_score(y_test, lr_pred, average='weighted')
lr_auc = roc_auc_score(
    y_test,
    lr_model.predict_proba(X_test),
    multi_class='ovr'
)


print(".........Decision Tree.........")
dt_model=DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train,y_train)
y_pred_dt=dt_model.predict(X_test)
dt_pred = dt_model.predict(X_test)

dt_acc = accuracy_score(y_test, dt_pred)
dt_pre = precision_score(y_test, dt_pred, average='weighted')
dt_rec = recall_score(y_test, dt_pred, average='weighted')
dt_f1 = f1_score(y_test, dt_pred, average='weighted')
dt_auc = roc_auc_score(
    y_test,
    dt_model.predict_proba(X_test),
    multi_class='ovr'
)

print(".........Random Forest.........")
rf_model=RandomForestClassifier(random_state=42)
rf_model.fit(X_train,y_train)
y_pred_rf=rf_model.predict(X_test)  
rf_pred = rf_model.predict(X_test)

rf_acc = accuracy_score(y_test, rf_pred)
rf_pre = precision_score(y_test, rf_pred, average='weighted')
rf_rec = recall_score(y_test, rf_pred, average='weighted')
rf_f1 = f1_score(y_test, rf_pred, average='weighted')
rf_auc = roc_auc_score(
    y_test,
    rf_model.predict_proba(X_test),
    multi_class='ovr'
)


print(".........XGBoost.........")
xgb_model=XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
xgb_model.fit(X_train,y_train)
y_pred_xgb=xgb_model.predict(X_test)
xgb_pred = xgb_model.predict(X_test)

xgb_acc = accuracy_score(y_test, xgb_pred)
xgb_pre = precision_score(y_test, xgb_pred, average='weighted')
xgb_rec = recall_score(y_test, xgb_pred, average='weighted')
xgb_f1 = f1_score(y_test, xgb_pred, average='weighted')
xgb_auc = roc_auc_score(
    y_test,
    xgb_model.predict_proba(X_test),
    multi_class='ovr'
)

import pandas as pd

comparison = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
        "XGBoost"
    ],
    "Accuracy": [lr_acc, dt_acc, rf_acc, xgb_acc],
    "Precision": [lr_pre, dt_pre, rf_pre, xgb_pre],
    "Recall": [lr_rec, dt_rec, rf_rec, xgb_rec],
    "F1-Score": [lr_f1, dt_f1, rf_f1, xgb_f1],
    "ROC-AUC": [lr_auc, dt_auc, rf_auc, xgb_auc]
})

print(comparison)
print(">.............................................................................................")
print("Starting Model Tuning & Optimization...")

param_grid = {
    'max_depth': [3, 5, 7, 9],
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [50, 100, 200],
    'subsample': [0.8, 1.0]
}

base_model = xgb_model

tuned_model = RandomizedSearchCV(
    estimator=base_model,
    param_distributions=param_grid,
    n_iter=10,
    scoring='accuracy',
    cv=3,
    verbose=1,
    random_state=42
)

tuned_model.fit(X, y)
best_xgb = tuned_model.best_estimator_

print(f"\nBest Parameters Found: {tuned_model.best_params_}")

joblib.dump(best_xgb, "src/authentication_model.pkl")
print("Optimized model saved successfully!")

print("\nCalculating Feature Importance & SHAP values...")

importance = best_xgb.feature_importances_
feat_imp_df = pd.DataFrame({'Feature': X.columns, 'Importance': importance}).sort_values(by='Importance', ascending=False)
print("\n--- Basic Feature Importance ---")
print(feat_imp_df)

explainer = shap.TreeExplainer(best_xgb)
shap_values = explainer.shap_values(X)

if len(shap_values.shape) == 3:
    shap_values = [shap_values[:, :, i] for i in range(shap_values.shape[2])]

plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X, show=False, plot_type="bar")
plt.tight_layout()
plt.savefig("src/shap_summary.png")
print("\nSHAP summary plot saved as 'src/shap_summary.png'!")