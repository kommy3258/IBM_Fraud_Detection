import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score, 
    precision_score, recall_score, f1_score, 
    classification_report
)
from tqdm import tqdm
import gc
from tslearn.barycenters import dtw_barycenter_averaging
from joblib import Parallel, delayed
import multiprocessing
from scipy.integrate import trapezoid

np.trapz = trapezoid
np.in1d = np.isin

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
WINDOW = 50
BATCH_SIZE = 8192
SHARD_SIZE = 2000000
TARGET_FRAUD_RATIO = 0.01

def generate_wdba_sample(real_patterns, window_size):
    np.trapz = trapezoid
    np.in1d = np.isin
    subset = real_patterns[np.random.choice(len(real_patterns), 5, replace=False)]
    main_weight = 0.8
    rest_weight = (1.0 - main_weight) / 4
    weights = np.array([main_weight] + [rest_weight] * 4, dtype=np.float32)
    return dtw_barycenter_averaging(subset, weights=weights, max_iter=2, verbose=False)

class FraudHybridDataset(Dataset):
    def __init__(self, data, labels, indices, window_size):
        self.data = data
        self.labels = labels
        self.indices = indices
        self.window_size = window_size
    def __len__(self): return len(self.indices)
    def __getitem__(self, idx):
        end_idx = self.indices[idx]
        x = self.data[end_idx - self.window_size + 1 : end_idx + 1]
        y = self.labels[end_idx]
        return torch.tensor(x, dtype=torch.float), torch.tensor(y, dtype=torch.float)

class AugmentDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x = torch.tensor(x_data, dtype=torch.float)
        self.y = torch.tensor(y_data, dtype=torch.float)
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.x[idx], self.y[idx]

class TransformerFraudModel(nn.Module):
    def __init__(self, cat_dims, num_dim, embed_dim=128, nhead=8, num_layers=3):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(dim, embed_dim) for dim in cat_dims])
        
        combined_dim = len(cat_dims) * embed_dim + num_dim
        self.input_projection = nn.Linear(combined_dim, embed_dim)
        
        self.pos_embedding = nn.Parameter(torch.zeros(1, WINDOW, embed_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=embed_dim * 4,
            dropout=0.2,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            
            nn.Linear(64, 1)
        )

    def forward(self, x):
        num_cat = len(self.embeddings)
        x_cat = x[:, :, :num_cat].long()
        x_num = x[:, :, num_cat:].float()
        
        embeds = [emb(x_cat[:, :, i]) for i, emb in enumerate(self.embeddings)]
        x_comb = torch.cat(embeds + [x_num], dim=-1)
        
        x_proj = self.input_projection(x_comb) + self.pos_embedding
        
        x_out = self.transformer_encoder(x_proj)
        
        last_step_out = x_out[:, -1, :]
        return self.classifier(last_step_out)

if __name__ == '__main__':
    print("Loading Data...")
    df = pd.read_csv("final.csv")
    df = df.sort_values(["User", "Card", "Year", "Month", "Day", "Hour", "Minute"]).reset_index(drop=True)
    cat_cols = ["Merchant Name"]
    time_cols = ["Year"]
    exclude_cols = []

    df['Global_Card_ID_Idx'] = LabelEncoder().fit_transform(df['User'].astype(str) + "_" + df['Card'].astype(str))
    num_cols = [c for c in df.columns if c not in cat_cols + ["User", "Card", "Is Fraud?", "Global_Card_ID_Idx"] + time_cols + exclude_cols]

    cat_dims = []
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        cat_dims.append(int(len(le.classes_)))

    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols].astype(np.float32))

    data_values = df[cat_cols + num_cols].values.astype(np.float32)
    labels = df["Is Fraud?"].values.astype(np.float32)
    card_ids = df['Global_Card_ID_Idx'].values

    valid_mask = (card_ids[WINDOW - 1:] == card_ids[:-WINDOW + 1])
    valid_indices = (np.where(valid_mask)[0] + (WINDOW - 1)).astype(np.int32)
    valid_years = df['Year'].values[valid_indices]

    train_indices = valid_indices[(valid_years >= 2000) & (valid_years <= 2017)]
    val_indices   = valid_indices[(valid_years >= 2018) & (valid_years <= 2018)]
    test_indices  = valid_indices[(valid_years == 2019)]

    print(f"\n[Split Info] Train: {len(train_indices):,} | Val: {len(val_indices):,} | Test: {len(test_indices):,}")

    fraud_indices = train_indices[labels[train_indices] == 1]
    normal_indices = train_indices[labels[train_indices] == 0]

    target_total_fraud = int(len(train_indices) * TARGET_FRAUD_RATIO)
    num_to_augment = max(0, target_total_fraud - len(fraud_indices))
    
    sample_fraud_idx = np.random.choice(fraud_indices, min(len(fraud_indices), 2000), replace=False)
    real_fraud_patterns = np.array([data_values[i-WINDOW+1 : i+1] for i in sample_fraud_idx])

    aug_x = Parallel(n_jobs=-1)(
        delayed(generate_wdba_sample)(real_fraud_patterns, WINDOW) 
        for _ in tqdm(range(num_to_augment), desc="wDBA Augmenting")
    )
    aug_dataset = AugmentDataset(np.array(aug_x, dtype=np.float32), np.ones(len(aug_x), dtype=np.float32))

    model = TransformerFraudModel(cat_dims, len(num_cols)).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    
    # FocalLoss 대신 nn.BCEWithLogitsLoss를 사용합니다.
    weight_val = len(normal_indices) / len(fraud_indices)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weight_val).to(DEVICE))

    for epoch in range(1, 11):
        sampled_normal = np.random.choice(normal_indices, min(len(normal_indices), SHARD_SIZE), replace=False)
        epoch_indices = np.concatenate([fraud_indices, sampled_normal])
        np.random.shuffle(epoch_indices)

        train_loader = DataLoader(
            ConcatDataset([FraudHybridDataset(data_values, labels, epoch_indices, WINDOW), aug_dataset]), 
            batch_size=BATCH_SIZE, shuffle=True
        )

        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for x_batch, y_batch in pbar:
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            logits = model(x_batch).squeeze()
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

    model.eval()
    test_loader = DataLoader(FraudHybridDataset(data_values, labels, test_indices, WINDOW), batch_size=BATCH_SIZE)
    test_probs, test_true = [], []
    
    with torch.no_grad():
        for x_t, y_t in tqdm(test_loader, desc="Testing"):
            x_t = x_t.to(DEVICE)
            logits = model(x_t).squeeze()
            test_probs.append(torch.sigmoid(logits).cpu().numpy())
            test_true.append(y_t.numpy())
            
    y_true = np.concatenate(test_true)
    y_prob = np.concatenate(test_probs)

    print(f"\nROC-AUC: {roc_auc_score(y_true, y_prob):.6f}")
    print(f"PR-AUC:  {average_precision_score(y_true, y_prob):.6f}")
    
    best_threshold = 0.5
    best_f1 = 0
    for t in np.linspace(0.1, 0.9, 17):
        f1 = f1_score(y_true, (y_prob > t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t
    
    print(f"Best Threshold: {best_threshold:.2f} | Best F1: {best_f1:.4f}")
    print(classification_report(y_true, (y_prob > best_threshold).astype(int)))