import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import (
    roc_auc_score, 
    average_precision_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    classification_report
)

# 1. 데이터 로드 및 전처리
df = pd.read_csv("C:/Users/LABKJH/Desktop/ipynb/card farud US/done.csv")

# 시간순 정렬 (데이터의 흐름을 유지하기 위함)
df = df.sort_values(["User", "Card", "unixtime"]).reset_index(drop=True)
df = df[df["Year"] >= 2000].reset_index(drop=True)

# 불필요한 컬럼 제거
df.drop(columns=["Merchant Name", "Merchant State", "Home_State", "last_off_State"], inplace=True)

# 2. 피처 및 라벨 분리
# User, Card, unixtime은 식별자이므로 학습에서 제외
EXCLUDE_COLS = ["User", "Card", "unixtime", "Is Fraud?"]
FEATURE_COLS = [c for c in df.columns if c not in EXCLUDE_COLS]

X = df[FEATURE_COLS]
y = df["Is Fraud?"]

# 3. 데이터 분할 (Time-based Split)
# 시계열 특성을 고려하여 뒤쪽 데이터를 테스트셋으로 사용
n = len(df)
train_end = int(n * 0.7)
val_end   = int(n * 0.9)

X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
X_val, y_val     = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
X_test, y_test   = X.iloc[val_end:], y.iloc[val_end:]

# 4. 불균형 처리를 위한 가중치 계산
# (음성 샘플 수 / 양성 샘플 수)
pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

# 5. 모델 정의 및 학습
model = XGBClassifier(
    scale_pos_weight=pos_weight,
    n_estimators=1000,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="auc",
    tree_method="hist"
)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=50
)

# 6. 평가
y_prob = model.predict_proba(X_test)[:, 1]
# 일반적인 분류 임계값 0.5 사용 (필요시 조정 가능)
y_pred = (y_prob > 0.6).astype(int)

print("\n### 최종 평가 결과 ###")
print(f"- ROC-AUC:     {roc_auc_score(y_test, y_prob):.6f}")
print(f"- PR-AUC:      {average_precision_score(y_test, y_prob):.6f}")
print(f"- Precision:   {precision_score(y_test, y_pred):.6f}")
print(f"- Recall:      {recall_score(y_test, y_pred):.6f}")
print(f"- F1-Score:    {f1_score(y_test, y_pred):.6f}")
print("\n[Classification Report]")
print(classification_report(y_test, y_pred))