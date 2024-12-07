import pandas as pd
import numpy as np
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from models.lstm import CustomLSTMCell, CustomLSTMModel, ModelManager, DatasetManager

lstm_model_path = "./ckpts/lstm_vae_1622_512.pth"
#lstm_model_path = './ckpts/test_bilstm256.pth'
lstm_model = CustomLSTMModel(input_dim=48, hidden_dim=512, output_dim=16)
lstm_model.load_model(model_path=lstm_model_path)

features = pd.read_csv('../data/processed/vae/features_vae_iv16_22_16.csv')

feature_cols = [f'feature_{i}' for i in range(16)]
for i in range(22,len(features)):
    ma1 = torch.tensor(features.iloc[i-1][feature_cols].astype(float).values, dtype=torch.float32).to(device)
    ma2 = torch.tensor(features.iloc[i-2][feature_cols].astype(float).values, dtype=torch.float32).to(device)
    ma3 = torch.tensor(features.iloc[i-3][feature_cols].astype(float).values, dtype=torch.float32).to(device)
    feature = torch.cat((ma1, ma2, ma3), dim=0).to(device)
    out = lstm_model.predict(feature)
    for j in range(16):
        features.at[i, f'feature_{j}'] = out[0][j].item()

features = features.dropna().reset_index(drop=True)

df_iv_path_list = [
    "../data/processed/pca/predicted_iv16.csv",
    "../data/processed/pca/predicted_iv17.csv",
    "../data/processed/pca/predicted_iv18.csv",
    "../data/processed/pca/predicted_iv19.csv",
    "../data/processed/pca/predicted_iv20.csv",
    "../data/processed/pca/predicted_iv21.csv",
    "../data/processed/pca/predicted_iv22.csv"
]

merged_df = pd.DataFrame()
for path in df_iv_path_list:
    df = pd.read_csv(path)
    merged_df = pd.concat([merged_df, df], axis=0)

merged_df = merged_df.reset_index(drop=True)
print(len(merged_df))

df = pd.merge(merged_df, features, on='date')

df = df[:30000]

from models.dnn import IVDataset, IVSDNN, train_model, large_moneyness_penalty, butterfly_arbitrage_penalty, calendar_spread_penalty, safe_divide

dataset = IVDataset(df, feature_cols)

from torch.utils.data import DataLoader
train_loader = DataLoader(dataset, batch_size=256, shuffle=True)
dnn = IVSDNN(input_size=dataset.get_input_size(), hidden_size=512)

import wandb
wandb.init(project="vae-dnn")
train_model(dnn, train_loader, 100, 0.001, 1, wandb)