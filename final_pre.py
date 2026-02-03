import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("credit_card_transactions-ibm_v2.csv")
zip_df = pd.read_csv('USZipsWithLatLon_20231227.csv')
city_df = pd.read_csv("worldcities.csv")
user_df = pd.read_csv("sd254_users_modified.csv")
card_df = pd.read_csv("sd254_cards_modified.csv")

df = df[(df["Year"] >= 2000) & (df["Year"] <= 2019)].reset_index(drop=True)

df['Hour'] = df['Time'].str.split(':').str[0].astype(int)
df['Minute'] = df['Time'].str.split(':').str[1].astype(int)
df['Amount'] = df['Amount'].str.replace('$', '').astype('float32')

df['Chip_Online'] = (df['Use Chip'] == 'Online Transaction').astype(int)
df['Chip_Swipe'] = (df['Use Chip'] == 'Swipe Transaction').astype(int)

df['Is Fraud?'] = (df['Is Fraud?'] == 'Yes').astype(int)

merchant_counts = df['Merchant Name'].value_counts()
df['Merchant Name'] = df['Merchant Name'].astype(str)
#low_freq_merchants = merchant_counts[merchant_counts < 244].index
#df.loc[df['Merchant Name'].isin(low_freq_merchants), 'Merchant Name'] = 'Others'

error_list = ['Insufficient Balance', 'Bad PIN', 'Technical Glitch', 'Bad Card Number', 'Bad CVV', 'Bad Expiration', 'Bad Zipcode']
for err in error_list:
    df[f'Error_{err}'] = df['Errors?'].str.contains(err, na=False).astype(int)

df = df.merge(zip_df[['zipcode', 'latitude', 'longitude']], how='left', left_on='Zip', right_on='zipcode')
df.loc[df['Merchant City'] == 'ONLINE', ['latitude', 'longitude']] = 0

df = df.merge(city_df[['city_ascii', 'country', 'lat', 'lng']], how='left', left_on=['Merchant City', 'Merchant State'], right_on=['city_ascii', 'country'])
df['Merchant_Lat'] = (df['lat'].combine_first(df['latitude'])).astype('float32')
df['Merchant_Lng'] = (df['lng'].combine_first(df['longitude'])).astype('float32')
df.drop(columns=['Time', 'Zip', 'zipcode', 'Errors?', 'city_ascii', 'country', 'lat', 'lng', 'Use Chip', 'latitude', 'longitude'], inplace=True)

df = df.merge(user_df, how='left', left_on='User', right_on='User_index')
df.drop(columns=['User_index', 'User_Current Age'], inplace=True)

df['Home_Lat'] = (df['Home_Latitude']).astype('float32')
df['Home_Lng'] = (df['Home_Longitude']).astype('float32')
df.drop(columns=['Home_Latitude', 'Home_Longitude'], inplace=True)

df = df.merge(card_df, how='left', left_on=['User', 'Card'], right_on=['Card_User', 'Card_Index'])
df.drop(columns=['Card_User', 'Card_Index'], inplace=True)

df['Card_Ex_Month'] = df['Card_Expires'].str.split('/').str[0].astype(int)
df['Card_Ex_Year'] = df['Card_Expires'].str.split('/').str[1].astype(int)
df.drop(columns=['Card_Expires'], inplace=True)

df['Card_Open_Month'] = df['Card_Open'].str.split('/').str[0].astype(int)
df['Card_Open_Year'] = df['Card_Open'].str.split('/').str[1].astype(int)
df.drop(columns=['Card_Open'], inplace=True)

df['Card_Brand_Master'] = (df['Card_Brand'] == 'Mastercard').astype(int)
df['Card_Brand_Amex'] = (df['Card_Brand'] == 'Amex').astype(int)
df['Card_Brand_Discover'] = (df['Card_Brand'] == 'Discover').astype(int)
df.drop(columns=['Card_Brand'], inplace=True)

df['Card_Type_Debit'] = (df['Card_Type'] == 'Debit').astype(int)
df['Card_Type_Credit'] = (df['Card_Type'] == 'Credit').astype(int)
df.drop(columns=['Card_Type', 'Merchant City'], inplace=True)

df.drop(columns=['Home_State', 'Merchant State'], inplace=True)
"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMB_DIM = 32
BATCH_SIZE = 2**19
EPOCHS = 10
LR = 0.05
le = LabelEncoder()
merchant_indices = le.fit_transform(df['Merchant Name']).astype(np.int64)
targets = df['Is Fraud?'].values.astype(np.float32)
X_gpu = torch.from_numpy(merchant_indices).to(device)
y_gpu = torch.from_numpy(targets).to(device).view(-1, 1)
class FastEmbeddingNet(nn.Module):
    def __init__(self, n_categories, emb_dim):
        super().__init__()
        self.embed = nn.Embedding(n_categories, emb_dim)
        self.fc = nn.Sequential(
            nn.Linear(emb_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = self.embed(x)
        return self.fc(x)
num_classes = len(le.classes_)
model = FastEmbeddingNet(num_classes, EMB_DIM).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
model.train()
n_samples = X_gpu.size(0)
for epoch in range(EPOCHS):
    perm = torch.randperm(n_samples).to(device)
    X_gpu = X_gpu[perm]
    y_gpu = y_gpu[perm]
    epoch_loss = 0
    for i in range(0, n_samples, BATCH_SIZE):
        batch_x = X_gpu[i : i + BATCH_SIZE]
        batch_y = y_gpu[i : i + BATCH_SIZE]
        optimizer.zero_grad()
        output = model(batch_x)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {epoch_loss/(n_samples/BATCH_SIZE):.4f}")
embedding_weights = model.embed.weight.detach().cpu().numpy()
emb_cols = [f'Merchant_Emb_{i}' for i in range(EMB_DIM)]
df_emb = pd.DataFrame(embedding_weights, columns=emb_cols)
df_emb['Merchant Name'] = le.classes_
df = df.merge(df_emb, on='Merchant Name', how='left')
df.drop(columns=['Merchant Name'], inplace=True)
del X_gpu, y_gpu, model
torch.cuda.empty_cache()
"""
df.sort_values(by=['User', 'Card', 'Year', 'Month', 'Day', 'Hour', 'Minute'], inplace=True)

df.to_csv("final.csv")