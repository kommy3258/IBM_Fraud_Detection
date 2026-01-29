import pandas as pd
import numpy as np
from tslearn.barycenters import dtw_barycenter_averaging
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, classification_report
from tqdm import tqdm

from sklearn.metrics import (
    roc_auc_score, 
    average_precision_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    classification_report,
    precision_recall_curve
)

if not hasattr(np, 'trapz'):
    from scipy.integrate import trapezoid
    np.trapz = trapezoid
if not hasattr(np, 'in1d'):
    np.in1d = np.isin

df = pd.read_csv("done.csv")
df = df.sort_values(["User", "Card", "unixtime"]).reset_index(drop=True)
df = df[df["Year"] >= 2000].reset_index(drop=True)
df.drop(columns=["Merchant Name", "Merchant State", "Home_State", "last_off_State"], inplace=True)

FEATURE_COLS = [c for c in df.columns 
                if c not in ["User", "Card", "unixtime", "Is Fraud?"]]

WINDOW = 20

def make_windows(group):
    X, y = [], []
    values = group[FEATURE_COLS].values
    labels = group["Is Fraud?"].values

    for i in range(len(group) - WINDOW + 1):
        x = values[i:i+WINDOW]
        label = int(labels[i:i+WINDOW].max() > 0)
        X.append(x)
        y.append(label)

    return X, y

X_all, y_all = [], []

for (_, _), g in df.groupby(["User", "Card"]):
    if len(g) >= WINDOW:
        X_tmp, y_tmp = make_windows(g)
        X_all.extend(X_tmp)
        y_all.extend(y_tmp)

X_all = np.array(X_all)
y_all = np.array(y_all)

n = len(X_all)
train_end = int(n * 0.7)
val_end   = int(n * 0.9)

X_train, y_train = X_all[:train_end], y_all[:train_end]
X_val, y_val     = X_all[train_end:val_end], y_all[train_end:val_end]
X_test, y_test   = X_all[val_end:], y_all[val_end:]

def wDBA_generate(X_fraud, n_samples):
    synthetic = []
    for _ in tqdm(range(n_samples)):
        idx = np.random.choice(len(X_fraud), 3, replace=False)
        bary = dtw_barycenter_averaging(X_fraud[idx])
        synthetic.append(bary)
    return np.array(synthetic)

fraud_ratio_target = 0.01
cur_ratio = y_train.mean()

if cur_ratio < fraud_ratio_target:
    X_fraud = X_train[y_train == 1]
    needed = int((fraud_ratio_target * len(y_train) - y_train.sum())
                 / (1 - fraud_ratio_target))

    X_syn = wDBA_generate(X_fraud, needed)
    y_syn = np.ones(len(X_syn))

    X_train = np.concatenate([X_train, X_syn])
    y_train = np.concatenate([y_train, y_syn])

X_train_flat = X_train.reshape(len(X_train), -1)
X_val_flat   = X_val.reshape(len(X_val), -1)
X_test_flat  = X_test.reshape(len(X_test), -1)

model = XGBClassifier(
    scale_pos_weight=10,
    n_estimators=1000,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="auc",
    tree_method="hist"
)

model.fit(
    X_train_flat, y_train,
    eval_set=[(X_val_flat, y_val)],
    verbose=50
)

y_prob = model.predict_proba(X_test_flat)[:, 1]
y_pred = (y_prob > 0.1).astype(int)

roc_auc = roc_auc_score(y_test, y_prob)
pr_auc = average_precision_score(y_test, y_prob)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"- ROC-AUC:    {roc_auc:.6f}")
print(f"- PR-AUC:     {pr_auc:.6f}")
print(f"- Precision:  {precision:.6f}")
print(f"- Recall:     {recall:.6f}")
print(f"- F1-Score:   {f1:.6f}")
print(classification_report(y_test, y_pred))