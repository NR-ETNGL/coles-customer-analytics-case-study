import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc

plt.rcParams.update({"font.size": 9})

def fit_and_roc(X, y, cat_cols, num_cols, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=random_state, stratify=y
    )
    preprocess = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ])
    curves = {}
    rf = Pipeline([("prep", preprocess), ("clf", RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_leaf=5,
        class_weight="balanced", random_state=random_state))])
    rf.fit(X_train, y_train)
    fpr, tpr, _ = roc_curve(y_test, rf.predict_proba(X_test)[:, 1])
    curves["Random Forest"] = (fpr, tpr, auc(fpr, tpr))

    svm = Pipeline([("prep", preprocess), ("clf", SVC(
        kernel="rbf", C=1.0, gamma="scale", probability=True,
        class_weight="balanced", random_state=random_state))])
    svm.fit(X_train, y_train)
    fpr, tpr, _ = roc_curve(y_test, svm.predict_proba(X_test)[:, 1])
    curves["SVM"] = (fpr, tpr, auc(fpr, tpr))
    return curves

retail = pd.read_csv("retail_rfm_features.csv")
num_cols_r = ["Recency", "AvgBasketValue", "UniqueProducts", "TotalItems"]
Xr = retail[num_cols_r + ["IsUK"]].copy()
Xr["IsUK"] = Xr["IsUK"].astype(str)
yr = retail["Engaged_Customer"]
curves_r = fit_and_roc(Xr, yr, ["IsUK"], num_cols_r)

cpa = pd.read_csv("cpa_features.csv")
cat_cols_c = ["Education", "Marital_Status"]
num_cols_c = [c for c in cpa.columns if c not in cat_cols_c + ["Response"]]
Xc = cpa[num_cols_c + cat_cols_c].copy()
yc = cpa["Response"]
curves_c = fit_and_roc(Xc, yc, cat_cols_c, num_cols_c)

fig, axes = plt.subplots(1, 2, figsize=(8, 3.6))
for name, (fpr, tpr, roc_auc) in curves_r.items():
    axes[0].plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.2f})")
axes[0].plot([0, 1], [0, 1], "k--", lw=0.8)
axes[0].set_title("Dataset 1: Online Retail\n(Engaged Customer)")
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].legend(fontsize=7, loc="lower right")

for name, (fpr, tpr, roc_auc) in curves_c.items():
    axes[1].plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.2f})")
axes[1].plot([0, 1], [0, 1], "k--", lw=0.8)
axes[1].set_title("Dataset 2: Customer Personality\n(Campaign Response)")
axes[1].set_xlabel("False Positive Rate")
axes[1].legend(fontsize=7, loc="lower right")

plt.tight_layout()
plt.savefig("roc_curves.png", dpi=200)
print("Saved roc_curves.png")

# ---- Feature importance figure (RF, both datasets) ----
from sklearn.inspection import permutation_importance

def rf_importance(X, y, cat_cols, num_cols, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=random_state, stratify=y)
    preprocess = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)])
    rf = Pipeline([("prep", preprocess), ("clf", RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_leaf=5,
        class_weight="balanced", random_state=random_state))])
    rf.fit(X_train, y_train)
    feat_names = num_cols + list(rf.named_steps["prep"].named_transformers_["cat"].get_feature_names_out(cat_cols))
    importances = rf.named_steps["clf"].feature_importances_
    return sorted(zip(feat_names, importances), key=lambda x: -x[1])[:6]

imp_r = rf_importance(Xr, yr, ["IsUK"], num_cols_r)
imp_c = rf_importance(Xc, yc, cat_cols_c, num_cols_c)

fig, axes = plt.subplots(1, 2, figsize=(8, 3.6))
axes[0].barh([f[0] for f in imp_r][::-1], [f[1] for f in imp_r][::-1], color="#4C72B0")
axes[0].set_title("Top features: Online Retail\n(Random Forest)")
axes[1].barh([f[0] for f in imp_c][::-1], [f[1] for f in imp_c][::-1], color="#DD8452")
axes[1].set_title("Top features: Customer Personality\n(Random Forest)")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=200)
print("Saved feature_importance.png")
