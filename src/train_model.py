import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
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


categorical_cols = [
     "location",
    "device_type",
    "browser",
    "login_method",
    "auth_type",
    "account_status",
    "role"
]
encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

joblib.dump(encoders, "src/label_encoders.pkl")
features = [
      "failed_attempts",
    "device_type",
    "browser",
    "location",
    "mfa_enabled",
    "login_method",
    "role"
]
df["blocked"] = df["blocked"].astype(int)
X = df[features]
y = df["blocked"]

X_train,X_test,y_train,y_test=train_test_split(
    X,
    y,
    random_state=42,
    test_size=0.2,
    stratify=y
)

from sklearn.preprocessing import StandardScaler

numeric_cols = [
    "failed_attempts"
]

scaler = StandardScaler()
X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

print(".........Logistic Regression.........")
lr_model=LogisticRegression(max_iter=1000)
lr_model.fit(X_train_sm, y_train_sm)
lr_pred = lr_model.predict(X_test)

lr_acc = accuracy_score(y_test, lr_pred)
lr_pre = precision_score(y_test, lr_pred, average='weighted')
lr_rec = recall_score(y_test, lr_pred, average='weighted')
lr_f1 = f1_score(y_test, lr_pred, average='weighted')
lr_auc = roc_auc_score(
    y_test,
    lr_model.predict_proba(X_test)[:, 1]
)


print("\nClassification Report - Logistic Regression")
print(classification_report(y_test, lr_pred))

print("\nConfusion Matrix - Logistic Regression")
print(confusion_matrix(y_test, lr_pred))
print(".........Decision Tree.........")
dt_model=DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train_sm, y_train_sm)
dt_pred = dt_model.predict(X_test)

dt_acc = accuracy_score(y_test, dt_pred)
dt_pre = precision_score(y_test, dt_pred, average='weighted')
dt_rec = recall_score(y_test, dt_pred, average='weighted')
dt_f1 = f1_score(y_test, dt_pred, average='weighted')
dt_auc = roc_auc_score(
    y_test,
    dt_model.predict_proba(X_test)[:, 1]
)


print("\nClassification Report - Decision Tree")
print(classification_report(y_test, dt_pred))

print("\nConfusion Matrix - Decision Tree")
print(confusion_matrix(y_test, dt_pred))
print(".........Random Forest.........")


rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    
)
rf_model.fit(X_train_sm, y_train_sm)
rf_pred = rf_model.predict(X_test)

rf_acc = accuracy_score(y_test, rf_pred)
rf_pre = precision_score(y_test, rf_pred, average='weighted')
rf_rec = recall_score(y_test, rf_pred, average='weighted')
rf_f1 = f1_score(y_test, rf_pred, average='weighted')
rf_auc = roc_auc_score(
    y_test,
    rf_model.predict_proba(X_test)[:, 1]
)


print("\nClassification Report - Random Forest")
print(classification_report(y_test, rf_pred))

print("\nConfusion Matrix - Random Forest")
print(confusion_matrix(y_test, rf_pred))
print(".........XGBoost.........")
xgb_model=XGBClassifier(eval_metric='mlogloss', random_state=42)
xgb_model.fit(X_train_sm, y_train_sm)
xgb_pred = xgb_model.predict(X_test)

xgb_acc = accuracy_score(y_test, xgb_pred)
xgb_pre = precision_score(y_test, xgb_pred, average='weighted')
xgb_rec = recall_score(y_test, xgb_pred, average='weighted')
xgb_f1 = f1_score(y_test, xgb_pred, average='weighted')
xgb_auc = roc_auc_score(
    y_test,
    xgb_model.predict_proba(X_test)[:, 1]   
)


print("\nClassification Report - XGBoost")
print(classification_report(y_test, xgb_pred))

print("\nConfusion Matrix - XGBoost")
print(confusion_matrix(y_test, xgb_pred))
print(".........CatBoost.........")
cb_model = CatBoostClassifier(random_state=42, verbose=False)
cb_model.fit(X_train_sm, y_train_sm)
cb_pred = cb_model.predict(X_test)

cb_acc = accuracy_score(y_test, cb_pred)
cb_pre = precision_score(y_test, cb_pred, average='weighted')
cb_rec = recall_score(y_test, cb_pred, average='weighted')
cb_f1 = f1_score(y_test, cb_pred, average='weighted')
cb_auc = roc_auc_score(
    y_test,
    cb_model.predict_proba(X_test)[:, 1]    
)


print("\nClassification Report - CatBoost")
print(classification_report(y_test, cb_pred))

print("\nConfusion Matrix - CatBoost")
print(confusion_matrix(y_test, cb_pred))
comparison = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
        "XGBoost",
        "CatBoost"
    ],
    "Accuracy": [lr_acc, dt_acc, rf_acc, xgb_acc, cb_acc],
    "Precision": [lr_pre, dt_pre, rf_pre, xgb_pre, cb_pre],
    "Recall": [lr_rec, dt_rec, rf_rec, xgb_rec, cb_rec],
    "F1-Score": [lr_f1, dt_f1, rf_f1, xgb_f1, cb_f1],
    "ROC-AUC": [lr_auc, dt_auc, rf_auc, xgb_auc, cb_auc]
})

comparison = comparison.sort_values(
    by="Accuracy",
    ascending=False
)

print(comparison)
print(">.............................................................................................")

print("Starting CatBoost Model Tuning & Optimization...")

param_grid = {
    "iterations": [100, 200, 300],
    "learning_rate": [0.01, 0.05, 0.1],
    "depth": [4, 6, 8, 10],
    "l2_leaf_reg": [1, 3, 5, 7, 9],
    "border_count": [32, 64, 128],
    "bagging_temperature": [0, 1, 3, 5]
}

tuned_model = RandomizedSearchCV(
    estimator=CatBoostClassifier(
        random_state=42,
        verbose=False
    ),
    param_distributions=param_grid,
    n_iter=10,
    scoring="f1_weighted",
    cv=3,
    random_state=42,
    verbose=1,
    n_jobs=-1
)

tuned_model.fit(X_train_sm, y_train_sm)

best_cb = tuned_model.best_estimator_

best_pred = best_cb.predict(X_test)

print("\nOptimized CatBoost")
print(classification_report(y_test, best_pred))

print("\nConfusion Matrix")
print(confusion_matrix(y_test, best_pred))

print("Accuracy:", accuracy_score(y_test, best_pred))
print("Precision:", precision_score(y_test, best_pred, average="weighted"))
print("Recall:", recall_score(y_test, best_pred, average="weighted"))
print("F1 Score:", f1_score(y_test, best_pred, average="weighted"))
print("ROC-AUC:", roc_auc_score(y_test, best_cb.predict_proba(X_test)[:, 1]))

print("\nBest Parameters:")
print(tuned_model.best_params_)

print("\nBest CV F1:")
print(tuned_model.best_score_)

joblib.dump(best_cb, "src/authentication_model.pkl")

print("Optimized model saved successfully!")



print("\nCalculating Feature Importance & SHAP values...")

importance = best_cb.feature_importances_

feat_imp_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance
}).sort_values(by="Importance", ascending=False)

print(feat_imp_df)

feat_imp_df.to_csv("src/feature_importance.csv", index=False)



explainer = shap.TreeExplainer(best_cb)

shap_values = explainer.shap_values(X_test)

if isinstance(shap_values, list):
    shap_values = shap_values[1]

print("SHAP values calculated successfully!")

shap.summary_plot(shap_values, X_test, show=False)
plt.savefig("src/shap_summary.png", dpi=300, bbox_inches="tight")
plt.close()

shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
plt.savefig("src/shap_bar.png", dpi=300, bbox_inches="tight")
plt.close()

print("SHAP plots saved successfully!")
