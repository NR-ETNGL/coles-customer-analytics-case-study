import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.inspection import permutation_importance
import json

RESULTS = {}

def run_pipeline(name, X, y, cat_cols, num_cols, random_state=42):
    print(f"\n{'='*70}\n{name}\n{'='*70}")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=random_state, stratify=y
    )

    preprocess = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ])

    results = {}
    fitted_models = {}

    # ---- Random Forest ----
    rf_pipe = Pipeline([
        ("prep", preprocess),
        ("clf", RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=5,
            class_weight="balanced", random_state=random_state
        )),
    ])
    rf_pipe.fit(X_train, y_train)
    y_pred = rf_pipe.predict(X_test)
    y_proba = rf_pipe.predict_proba(X_test)[:, 1]
    results["Random Forest"] = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }
    fitted_models["Random Forest"] = rf_pipe

    # ---- SVM ----
    svm_pipe = Pipeline([
        ("prep", preprocess),
        ("clf", SVC(kernel="rbf", C=1.0, gamma="scale",
                     probability=True, class_weight="balanced",
                     random_state=random_state)),
    ])
    svm_pipe.fit(X_train, y_train)
    y_pred = svm_pipe.predict(X_test)
    y_proba = svm_pipe.predict_proba(X_test)[:, 1]
    results["SVM"] = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }
    fitted_models["SVM"] = svm_pipe

    for m, r in results.items():
        print(f"\n{m}:")
        for k, v in r.items():
            print(f"  {k}: {v:.3f}")

    # ---- Feature importance (RF native; permutation for SVM) ----
    feat_names = (
        num_cols +
        list(fitted_models["Random Forest"].named_steps["prep"]
             .named_transformers_["cat"].get_feature_names_out(cat_cols))
    )
    rf_importances = fitted_models["Random Forest"].named_steps["clf"].feature_importances_
    rf_imp_sorted = sorted(zip(feat_names, rf_importances), key=lambda x: -x[1])[:10]

    perm = permutation_importance(
        fitted_models["SVM"], X_test, y_test, n_repeats=8,
        random_state=random_state, scoring="roc_auc"
    )
    svm_imp_sorted = sorted(zip(X_test.columns, perm.importances_mean), key=lambda x: -x[1])[:10]

    print("\nTop RF feature importances:")
    for f, v in rf_imp_sorted:
        print(f"  {f}: {v:.4f}")
    print("\nTop SVM permutation importances (raw input columns):")
    for f, v in svm_imp_sorted:
        print(f"  {f}: {v:.4f}")

    RESULTS[name] = {
        "metrics": results,
        "rf_top_features": rf_imp_sorted,
        "svm_top_features": svm_imp_sorted,
        "n_train": len(X_train), "n_test": len(X_test),
        "pos_rate": float(y.mean()),
    }
    return fitted_models


# ============ Dataset 1: Online Retail (proxy target) ============
# NOTE: Engaged_Customer was defined directly from Frequency & Monetary
# (median split), so those two columns are excluded from the feature set
# to avoid leaking the label definition into the predictors.
retail = pd.read_csv("retail_rfm_features.csv")
num_cols_r = ["Recency", "AvgBasketValue", "UniqueProducts", "TotalItems"]
cat_cols_r = ["IsUK"]
Xr = retail[num_cols_r + cat_cols_r].copy()
Xr["IsUK"] = Xr["IsUK"].astype(str)
yr = retail["Engaged_Customer"]
run_pipeline("Dataset 1: Online Retail (target = Engaged_Customer proxy)",
             Xr, yr, cat_cols_r, num_cols_r)

# ============ Dataset 2: Customer Personality Analysis ============
cpa = pd.read_csv("cpa_features.csv")
cat_cols_c = ["Education", "Marital_Status"]
num_cols_c = [c for c in cpa.columns if c not in cat_cols_c + ["Response"]]
Xc = cpa[num_cols_c + cat_cols_c].copy()
yc = cpa["Response"]
run_pipeline("Dataset 2: Customer Personality Analysis (target = Response)",
             Xc, yc, cat_cols_c, num_cols_c)

with open("results_summary.json", "w") as f:
    json.dump(RESULTS, f, indent=2, default=float)
print("\n\nSaved results_summary.json")
