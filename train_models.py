"""
PRG4206 starter: Student Dropout & Academic Success
Run:  python train_models.py
Needs: data/dataset.csv  (put your downloaded file there)
Makes: figures/*.png, results.csv, models/best_model.joblib
"""
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             roc_auc_score, confusion_matrix,
                             ConfusionMatrixDisplay, classification_report)
from xgboost import XGBClassifier

SEED = 42
for d in ("figures", "models"):
    os.makedirs(d, exist_ok=True)

# ---------------------------------------------------------------- 1. LOAD
df = pd.read_csv("data/dataset.csv")
print("Shape:", df.shape)
print("Missing values:", df.isnull().sum().sum(), "| Duplicates:", df.duplicated().sum())

# ---------------------------------------------------------------- 2. EDA
# Class balance (Enrolled is the minority class: ~18%)
plt.figure(figsize=(5, 4))
df["Target"].value_counts().plot(kind="bar", color=["#4C72B0", "#C44E52", "#55A868"])
plt.title("Class distribution"); plt.ylabel("Students"); plt.xticks(rotation=0)
plt.tight_layout(); plt.savefig("figures/01_class_balance.png", dpi=200); plt.close()

# Correlation of numeric features with the target (encoded 0/1/2)
le = LabelEncoder()
y = le.fit_transform(df["Target"])          # Dropout=0, Enrolled=1, Graduate=2
X = df.drop(columns="Target")
corr = X.assign(Target=y).corr()["Target"].drop("Target").sort_values()
plt.figure(figsize=(7, 8))
top = pd.concat([corr.head(8), corr.tail(8)])
top.plot(kind="barh", color="#4C72B0")
plt.title("Top correlations with target (0=Dropout, 2=Graduate)")
plt.tight_layout(); plt.savefig("figures/02_target_correlation.png", dpi=200); plt.close()

# ---------------------------------------------------------------- 3. SPLIT
# 70 / 15 / 15, stratified so every split keeps the same class proportions
X_train, X_tmp, y_train, y_tmp = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=SEED)
X_val, X_test, y_val, y_test = train_test_split(
    X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=SEED)
print(f"Train {len(X_train)} | Val {len(X_val)} | Test {len(X_test)}")

# ---------------------------------------------------------------- 4. MODELS
models = {
    "Logistic Regression": make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED)),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, class_weight="balanced", random_state=SEED, n_jobs=-1),
    "XGBoost": XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, eval_metric="mlogloss", random_state=SEED),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
rows = []
fitted = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    fitted[name] = model

    def score(Xs, ys):
        pred = model.predict(Xs)
        proba = model.predict_proba(Xs)
        p, r, f, _ = precision_recall_fscore_support(ys, pred, average="macro", zero_division=0)
        return accuracy_score(ys, pred), p, r, f, roc_auc_score(ys, proba, multi_class="ovr")

    val_acc = score(X_val, y_val)[0]
    acc, p, r, f1, auc = score(X_test, y_test)
    cv_acc = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy").mean()
    rows.append(dict(Model=name, CV_Acc=cv_acc, Val_Acc=val_acc, Test_Acc=acc,
                     Precision=p, Recall=r, F1_macro=f1, ROC_AUC=auc))
    print(f"\n=== {name} ===")
    print(classification_report(y_test, model.predict(X_test), target_names=le.classes_))

    ConfusionMatrixDisplay(confusion_matrix(y_test, model.predict(X_test)),
                           display_labels=le.classes_).plot(cmap="Blues", colorbar=False)
    plt.title(f"Confusion matrix: {name}")
    plt.tight_layout()
    plt.savefig(f"figures/03_cm_{name.replace(' ', '_')}.png", dpi=200); plt.close()

results = pd.DataFrame(rows).round(4)
results.to_csv("results.csv", index=False)
print("\n", results.to_string(index=False))

# ---------------------------------------------------------------- 5. SAVE BEST
best_name = results.sort_values("F1_macro", ascending=False).iloc[0]["Model"]
print("\nBest model by macro-F1:", best_name)
joblib.dump({"model": fitted[best_name], "classes": list(le.classes_),
             "columns": list(X.columns)}, "models/best_model.joblib")

# ---------------------------------------------------------------- 6. SHAP
# SHAP on XGBoost (tree model, fast and clean). Explains which features push
# a student towards Dropout / Enrolled / Graduate.
xgb = fitted["XGBoost"]
explainer = shap.TreeExplainer(xgb)
sample = X_test.sample(n=min(300, len(X_test)), random_state=SEED)
sv = explainer.shap_values(sample)
sv = np.array(sv)
# handle both older (list of arrays) and newer (n, features, classes) shapes
dropout_sv = sv[0] if sv.ndim == 3 and sv.shape[0] == 3 else sv[:, :, 0]

plt.figure()
shap.summary_plot(dropout_sv, sample, show=False, max_display=15)
plt.title("SHAP: what pushes a student towards DROPOUT")
plt.tight_layout(); plt.savefig("figures/04_shap_dropout.png", dpi=200, bbox_inches="tight"); plt.close()

plt.figure()
mean_abs = np.abs(sv).mean(axis=(0, 2)) if sv.ndim == 3 and sv.shape[-1] == 3 else np.abs(sv).mean(axis=(0, 1))
pd.Series(mean_abs, index=X.columns).sort_values().tail(15).plot(kind="barh", color="#4C72B0")
plt.title("SHAP global importance (all classes)")
plt.tight_layout(); plt.savefig("figures/05_shap_importance.png", dpi=200); plt.close()

print("\nDone. Check the figures/ folder and results.csv")
