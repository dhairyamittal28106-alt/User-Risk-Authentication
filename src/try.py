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
    "auth_type",
    "account_status",
    "session_duration",
    "password_age_days",
    "role",
    "privilege_level",
    "suspicious_activity",
    "token_expired"
]
df["blocked"] = df["blocked"].astype(int)
X = df[features]
y = df["blocked"]


#train-test split
X_train,X_test,y_train,y_test=train_test_split(
    X,
    y,
    random_state=42,
    test_size=0.2,
    stratify=y
)

from sklearn.preprocessing import StandardScaler

numeric_cols = [
    "failed_attempts",
    "session_duration",
    "password_age_days",
    "privilege_level"
]

scaler = StandardScaler()
X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])
# Apply SMOTE
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

#Starting model training
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