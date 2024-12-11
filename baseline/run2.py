import pandas as pd
import numpy as np
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from models.lstm import CustomLSTMCell, CustomLSTMModel, ModelManager, DatasetManager

lstm_model_path = "./ckpts/lstm_vae_1622_512.pth"
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
print(f"Total samples: {len(merged_df)}")

df = pd.merge(merged_df, features, on='date')
df = df[:30000]
df_val = df[:-10000]

# Import the normalized dataset
from models.dnn import NormalizedIVDataset
dataset = NormalizedIVDataset(df, feature_cols)
val_dataset = NormalizedIVDataset(df_val,feature_cols)

from torch.utils.data import DataLoader
# Increase batch size for more stable training
train_loader = DataLoader(dataset, batch_size=256, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=True)

from models.dnn import IVSDNN, train_model

# Adjust model architecture
dnn = IVSDNN(
    input_size=dataset.get_input_size(),
    hidden_size=512 
)

# Initialize wandb
import wandb
wandb.init(project="vae-dnn", config={
    "hidden_size": 256,
    "batch_size": 512,
    "learning_rate": 0.001,
    "num_epochs": 100
})

# Train with modified hyperparameters
train_model(
    model=dnn,
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=100,
    learning_rate=0.001,
    lambda_penalty=1.0,
    wandb=wandb
)