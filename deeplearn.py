import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import gc
import warnings
warnings.filterwarnings("ignore")

class FraudDetectionModel(nn.Module):
    def __init__(self, user_dim, feat_dims, cont_dim, emb_dim=32):
        super().__init__()

        self.user_emb = nn.Embedding(user_dim, emb_dim)

        self.feat_embs = nn.ModuleDict({
            col: nn.Embedding(size, emb_dim)
            for col, size in feat_dims.items()
        })

        total_dim = emb_dim * (1 + len(feat_dims)) + cont_dim

        self.net = nn.Sequential(
            nn.Linear(total_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, user, cont, feats):
        u = self.user_emb(user)
        f = [self.feat_embs[k](feats[k]) for k in self.feat_embs]
        x = torch.cat([u] + f + [cont], dim=1)
        return self.net(x)

@torch.no_grad()
def compute_metrics_gpu(y_true, y_logit):
    threshold = 0.5
    y_prob = torch.sigmoid(y_logit)
    y_pred = (y_prob > threshold).float()
    eps = 1e-8

    tp = (y_pred * y_true).sum()
    fp = (y_pred * (1 - y_true)).sum()
    fn = ((1 - y_pred) * y_true).sum()
    tn = ((1 - y_pred) * (1 - y_true)).sum()

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    acc = (tp + tn) / (tp + tn + fp + fn + eps)

    prob_sorted, idx = torch.sort(y_prob, descending=True)
    true_sorted = y_true[idx]

    tps = torch.cumsum(true_sorted, dim=0)
    fps = torch.cumsum(1 - true_sorted, dim=0)

    tpr = tps / (tps[-1] + eps)
    fpr = fps / (fps[-1] + eps)
    roc = torch.trapz(tpr, fpr)

    precision_curve = tps / (tps + fps + eps)
    recall_curve = tps / (tps[-1] + eps)
    pr = torch.trapz(precision_curve, recall_curve)

    return {
        "acc": acc.item(),
        "precision": precision.item(),
        "recall": recall.item(),
        "f1": f1.item(),
        "roc": roc.item(),
        "pr": pr.item()
    }

def main():
    device = torch.device("cuda")
    df = pd.read_csv("done.csv")

    target_col = "Is Fraud?"
    user_col = "User"

    feature_cols = ["Card", "Merchant Name", "Merchant State", "Home_State", "last_off_State"]

    cont_cols = [c for c in df.columns if c not in [target_col, user_col] + feature_cols]

    encoders = {}
    for col in [user_col] + feature_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    scaler = StandardScaler()
    df[cont_cols] = scaler.fit_transform(df[cont_cols])

    train_df, val_df = train_test_split(
        df, test_size=0.2, stratify=df[target_col], random_state=42
    )

    train_u = torch.tensor(train_df[user_col].values).long().cuda()
    val_u   = torch.tensor(val_df[user_col].values).long().cuda()

    train_c = torch.tensor(train_df[cont_cols].values).float().cuda()
    val_c   = torch.tensor(val_df[cont_cols].values).float().cuda()

    train_f = {c: torch.tensor(train_df[c].values).long().cuda() for c in feature_cols}
    val_f   = {c: torch.tensor(val_df[c].values).long().cuda() for c in feature_cols}

    train_t = torch.tensor(train_df[target_col].values).float().cuda()
    val_t   = torch.tensor(val_df[target_col].values).float().cuda()

    del df, train_df, val_df
    gc.collect()

    model = FraudDetectionModel(
        user_dim=train_u.max().item() + 1,
        feat_dims={c: int(train_f[c].max()) + 1 for c in feature_cols},
        cont_dim=len(cont_cols),
        emb_dim=32
    ).cuda()

    optimizer = optim.Adam(model.parameters(), lr=0.003)
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.cuda.amp.GradScaler()

    batch_size = 131072
    epochs = 500

    for epoch in range(epochs):
        model.train()
        idx = torch.randperm(len(train_u), device="cuda")

        total_loss = 0.0

        for i in tqdm(range(0, len(idx), batch_size)):
            b = idx[i:i+batch_size]

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast():
                out = model(
                    train_u[b],
                    train_c[b],
                    {k: v[b] for k, v in train_f.items()}
                ).squeeze()

                loss = criterion(out, train_t[b])

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        model.eval()

        logits = []
        targets = []

        with torch.no_grad():
            for i in range(0, len(val_u), batch_size):
                with torch.cuda.amp.autocast():
                    logit = model(
                        val_u[i:i+batch_size],
                        val_c[i:i+batch_size],
                        {k: v[i:i+batch_size] for k, v in val_f.items()}
                    ).squeeze()

                logits.append(logit)
                targets.append(val_t[i:i+batch_size])

        y_logit = torch.cat(logits)
        y_true = torch.cat(targets)

        metrics = compute_metrics_gpu(y_true, y_logit)

        print(
            f"Epoch {epoch+1} | "
            f"Loss {total_loss:.6f} | "
            f"ACC {metrics['acc']:.6f} | "
            f"Precision {metrics['precision']:.6f} | "
            f"Recall {metrics['recall']:.6f} | "
            f"F1 {metrics['f1']:.6f} | "
            f"ROC {metrics['roc']:.6f} | "
            f"PR {metrics['pr']:.6f}"
        )
        print('='*70)

        del logits, targets, y_logit, y_true
        torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    main()
