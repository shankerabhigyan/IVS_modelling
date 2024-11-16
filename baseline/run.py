import pandas as pd
import numpy as np
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from models.lstm import CustomLSTMCell, CustomLSTMModel, ModelManager, DatasetManager

lstm_model_path = "./ckpts/lstm1620_256.pth"
#lstm_model_path = './ckpts/test_bilstm256.pth'
lstm_model = CustomLSTMModel(input_dim=9, hidden_dim=256, output_dim=3)
lstm_model.load_model(model_path=lstm_model_path)

features = pd.read_csv("../data/processed/features_pca_iv16-20.csv")

for i in range(22,len(features)):
    ma1 = torch.tensor(features.iloc[i-1][['feature1', 'feature2', 'feature3']].astype(float).values, dtype=torch.float32)
    ma5 = torch.tensor(features.iloc[i-5:i][['feature1', 'feature2', 'feature3']].mean(axis=0).values, dtype=torch.float32)
    ma22 = torch.tensor(features.iloc[i-22:i][['feature1', 'feature2', 'feature3']].mean(axis=0).values, dtype=torch.float32)
    feature = torch.cat((ma1, ma5, ma22), dim=0).to(device)
    out = lstm_model.predict(feature)
    for obj in out:
        features.at[i, "F1"] = obj[0].item()
        features.at[i, "F2"] = obj[1].item()
        features.at[i, "F3"] = obj[2].item()

        
features = features.dropna().reset_index(drop=True)

df_iv_path_list = [
    "../data/processed/pca/predicted_iv16.csv",
    "../data/processed/pca/predicted_iv17.csv",
    "../data/processed/pca/predicted_iv18.csv",
    "../data/processed/pca/predicted_iv19.csv",
    "../data/processed/pca/predicted_iv20.csv"
]

merged_df = pd.DataFrame()
for path in df_iv_path_list:
    df = pd.read_csv(path)
    merged_df = pd.concat([merged_df, df], axis=0)

merged_df = merged_df.reset_index(drop=True)
print(len(merged_df))

# join the two dataframes using the date column so that we have the corresponding F1, F2, F3 values for each date
df = pd.merge(merged_df, features, on='date')
df = df[:30000]
feature_cols = ['F1', 'F2', 'F3']
from models.dnn import IVDataset, IVSDNN, train_model, large_moneyness_penalty, butterfly_arbitrage_penalty, calendar_spread_penalty, safe_divide

dataset = IVDataset(df, feature_cols)

print(dataset.get_input_size())

from torch.utils.data import DataLoader
train_loader = DataLoader(dataset, batch_size=200, shuffle=True)
dnn = IVSDNN(input_size=dataset.get_input_size(), hidden_size=256)

import wandb
wandb.init(project="ivs-dnn")
train_model(dnn, train_loader, 100, 0.001, 1, wandb)