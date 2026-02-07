import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 8192
SHARD_SIZE = 5000000
TARGET_FRAUD_RATIO = 0.00
EPOCH = 10

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=3.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt)**self.gamma * BCE_loss
        if self.reduction == 'mean': return torch.mean(F_loss)
        elif self.reduction == 'sum': return torch.sum(F_loss)
        else: return F_loss

class GPUDataset(Dataset):
    def __init__(self, data_tensor, label_tensor, indices):
        self.data = data_tensor
        self.labels = label_tensor
        self.indices = indices

    def __len__(self): 
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        return self.data[real_idx], self.labels[real_idx]

class TabularTransformerModel(nn.Module):
    def __init__(self, cat_dims, num_dim, embed_dim=128, nhead=4, num_layers=3):
        super().__init__()
        self.cat_embeddings = nn.ModuleList([nn.Embedding(dim, embed_dim) for dim in cat_dims])
        self.num_projections = nn.ModuleList([nn.Linear(1, embed_dim) for _ in range(num_dim)])
        
        self.num_cat = len(cat_dims)
        self.num_num = num_dim
        total_tokens = self.num_cat + self.num_num
        
        self.pos_embedding = nn.Parameter(torch.zeros(1, total_tokens, embed_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead, dim_feedforward=embed_dim * 4,
            dropout=0.2, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * total_tokens, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        batch_size = x.size(0)
        cat_x = x[:, :self.num_cat].long()
        cat_embeds = [emb(cat_x[:, i]) for i, emb in enumerate(self.cat_embeddings)]
        
        num_x = x[:, self.num_cat:].float()
        num_embeds = [proj(num_x[:, i].unsqueeze(1)) for i, proj in enumerate(self.num_projections)]
        
        x_stack = torch.stack(cat_embeds + num_embeds, dim=1)
        x_proj = x_stack + self.pos_embedding
        x_out = self.transformer_encoder(x_proj)
        
        return self.classifier(x_out.reshape(batch_size, -1))

if __name__ == '__main__':
    print("Reading Data...")
    df = pd.read_csv("final.csv")
    
    cat_cols = ["Merchant Name"]
    time_cols = ["Year"]
    num_cols = [c for c in df.columns if c not in cat_cols + ["User", "Card", "Is Fraud?"] + time_cols]

    cat_dims = []
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        cat_dims.append(int(len(le.classes_)))

    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols].astype(np.float32))

    full_data_gpu = torch.tensor(df[cat_cols + num_cols].values, dtype=torch.float32).to(DEVICE)
    full_labels_gpu = torch.tensor(df["Is Fraud?"].values, dtype=torch.float32).to(DEVICE)
    years = df['Year'].values

    all_indices = np.arange(len(df))
    train_indices = all_indices[(years >= 2000) & (years <= 2017)]
    val_indices = all_indices[(years >= 2018) & (years <= 2018)]
    test_indices  = all_indices[(years >= 2019) & (years <= 2019)]

    fraud_indices = train_indices[full_labels_gpu[train_indices].cpu().numpy() == 1]
    normal_indices = train_indices[full_labels_gpu[train_indices].cpu().numpy() == 0]

    target_total_fraud = int(len(train_indices) * TARGET_FRAUD_RATIO)
    num_to_augment = max(0, target_total_fraud - len(fraud_indices))
    
    aug_data_gpu = None
    if num_to_augment > 0:
        print(f"Generating {num_to_augment} augmented samples...")
        real_fraud = full_data_gpu[fraud_indices]
        idx1 = torch.randint(0, len(real_fraud), (num_to_augment,))
        idx2 = torch.randint(0, len(real_fraud), (num_to_augment,))
        lambd = torch.distributions.Beta(0.5, 0.5).sample((num_to_augment, 1)).to(DEVICE)
        
        aug_data_gpu = lambd * real_fraud[idx1] + (1 - lambd) * real_fraud[idx2]
        aug_labels_gpu = torch.ones(num_to_augment, dtype=torch.float32).to(DEVICE)

    model = TabularTransformerModel(cat_dims, len(num_cols)).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    criterion = FocalLoss(alpha=0.25, gamma=2.0)

    for epoch in range(1, EPOCH + 1):
        sampled_normal = np.random.choice(normal_indices, min(len(normal_indices), SHARD_SIZE), replace=False)
        epoch_indices = np.concatenate([fraud_indices, sampled_normal])
        np.random.shuffle(epoch_indices)

        train_ds = GPUDataset(full_data_gpu, full_labels_gpu, epoch_indices)
        
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for x_batch, y_batch in pbar:
            
            optimizer.zero_grad()
            logits = model(x_batch).squeeze()
            
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

    model.eval()
    test_ds = GPUDataset(full_data_gpu, full_labels_gpu, test_indices)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)
    
    test_probs, test_true = [], []
    with torch.no_grad():
        for x_t, y_t in tqdm(test_loader, desc="Testing"):
            logits = model(x_t).squeeze()
            test_probs.append(torch.sigmoid(logits).cpu().numpy())
            test_true.append(y_t.cpu().numpy())
    
    y_true = np.concatenate(test_true)
    y_prob = np.concatenate(test_probs)
    print(f"\nROC-AUC: {roc_auc_score(y_true, y_prob):.6f}")
    print(f"\nPR-AUC: {average_precision_score(y_true, y_prob):.6f}")