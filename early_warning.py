"""
Early-warning experiment
Question: how early can we predict dropout?
  Phase A = data known at enrollment only
  Phase B = Phase A + 1st semester results
  Phase C = Phase B + 2nd semester results (everything; same as train_models.py)

Run:  python early_warning.py
Needs: data/dataset.csv
Makes: results_early_warning.csv, figures/06_early_warning.png
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score
from xgboost import XGBClassifier

SEED = 42
os.makedirs("figures", exist_ok=True)

df = pd.read_csv("data/dataset.csv")
le = LabelEncoder()
y = le.fit_transform(df["Target"])          # Dropout=0, Enrolled=1, Graduate=2
X = df.drop(columns="Target")

# Same 70/15/15 split and seed as train_models.py, so Phase C matches those results
X_train, X_tmp, y_train, y_tmp = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=SEED)
X_val, X_test, y_val, y_test = train_test_split(
    X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=SEED)

# Which columns belong to which phase
first_sem = [c for c in X.columns if c.startswith("Curricular units 1st sem")]
second_sem = [c for c in X.columns if c.startswith("Curricular units 2nd sem")]
enrollment = [c for c in X.columns if c not in first_sem + second_sem]

phases = {
    "A: Enrollment only": enrollment,
    "B: + 1st semester": enrollment + first_sem,
    "C: + 2nd semester (all)": enrollment + first_sem + second_sem,
}
print({k: len(v) for k, v in phases.items()}, "features per phase")

rows = []
for phase, cols in phases.items():
    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=SEED, n_jobs=-1),
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, eval_metric="mlogloss", random_state=SEED),
    }
    for name, model in models.items():
        model.fit(X_train[cols], y_train)
        pred = model.predict(X_test[cols])
        rows.append({
            "Phase": phase, "Model": name, "Features": len(cols),
            "Accuracy": accuracy_score(y_test, pred),
            "Macro_F1": f1_score(y_test, pred, average="macro"),
            "Dropout_recall": recall_score(y_test, pred, labels=[0], average=None)[0],
            "Enrolled_F1": f1_score(y_test, pred, labels=[1], average=None)[0],
        })

res = pd.DataFrame(rows).round(4)
res.to_csv("results_early_warning.csv", index=False)
print("\n", res.to_string(index=False))

# Chart: macro-F1 and Dropout recall per phase
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, metric, title in zip(axes, ["Macro_F1", "Dropout_recall"],
                             ["Macro F1", "Recall of the Dropout class"]):
    pivot = res.pivot(index="Phase", columns="Model", values=metric).reindex(list(phases))
    pivot.plot(kind="bar", ax=ax, color=["#55A868", "#4C72B0"])
    ax.set_title(title); ax.set_xlabel(""); ax.set_ylim(0.4, 1.0)
    ax.set_xticklabels([p.split(":")[0] for p in pivot.index], rotation=0)
axes[0].set_xlabel("A = enrollment only, B = + 1st semester, C = + 2nd semester")
plt.tight_layout()
plt.savefig("figures/06_early_warning.png", dpi=200)
plt.close()
print("\nSaved results_early_warning.csv and figures/06_early_warning.png")
