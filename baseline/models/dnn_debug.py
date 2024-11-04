import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from datetime import datetime

class IVDataset(Dataset):
    def __init__(self, features_df, iv_df):
        # Convert dates to datetime
        features_df['date'] = pd.to_datetime(features_df['date'])
        iv_df['date'] = pd.to_datetime(iv_df['date'])
        
        # Get unique dates
        feature_dates = set(features_df['date'])
        iv_dates = set(iv_df['date'])
        common_dates = sorted(list(feature_dates.intersection(iv_dates)))
        
        print(f"\nDate ranges:")
        print(f"Features: {min(feature_dates)} to {max(feature_dates)}")
        print(f"IV data: {min(iv_dates)} to {max(iv_dates)}")
        print(f"Common dates: {len(common_dates)}")
        
        # Create clean dataframes
        data_list = []
        feature_cols = [col for col in features_df.columns if col != 'date']
        
        for date in common_dates:
            # Get features for this date
            feat_row = features_df[features_df['date'] == date][feature_cols].iloc[0]
            
            # Get all IV points for this date
            iv_rows = iv_df[iv_df['date'] == date]
            
            # Create rows with features repeated for each IV point
            for _, iv_row in iv_rows.iterrows():
                row_data = {
                    'date': date,
                    'm': iv_row['m'],
                    'tau': iv_row['tau'],
                    'IV': iv_row['IV']
                }
                # Add features
                for col in feature_cols:
                    row_data[col] = feat_row[col]
                
                data_list.append(row_data)
        
        # Create final dataframe
        self.data = pd.DataFrame(data_list)
        
        print("\nDataset creation complete:")
        print(f"Total rows: {len(self.data)}")
        print("\nFeature stats:")
        print(self.data[feature_cols].describe())
        
        # Convert to tensors
        self.feature_cols = feature_cols
        self.features = torch.tensor(self.data[feature_cols].values, dtype=torch.float32)
        self.m = torch.tensor(self.data['m'].values, dtype=torch.float32).reshape(-1, 1)
        self.tau = torch.tensor(self.data['tau'].values, dtype=torch.float32).reshape(-1, 1)
        self.iv = torch.tensor(self.data['IV'].values, dtype=torch.float32).reshape(-1, 1)
        
        print("\nTensor shapes:")
        print(f"Features: {self.features.shape}")
        print(f"m: {self.m.shape}")
        print(f"tau: {self.tau.shape}")
        print(f"iv: {self.iv.shape}")
        
        # Verify no NaN values
        print("\nChecking for NaN values:")
        print(f"Features NaN: {torch.isnan(self.features).any()}")
        print(f"m NaN: {torch.isnan(self.m).any()}")
        print(f"tau NaN: {torch.isnan(self.tau).any()}")
        print(f"iv NaN: {torch.isnan(self.iv).any()}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_tensor = torch.cat([self.features[idx], self.m[idx], self.tau[idx]], dim=0)
        return input_tensor, self.iv[idx]
    
    def get_input_size(self):
        return len(self.feature_cols) + 2

class IVSDNN(nn.Module):
    def __init__(self, input_size, hidden_size=50):
        super(IVSDNN, self).__init__()
        self.input_size = input_size
        self.feature_size = input_size - 2
        
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Softplus()
        )
        
        # xavier initialisation
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
        print("\nModel parameter stats:")
        for name, param in self.net.named_parameters():
            print(f"{name}: min={param.min().item():.4f}, max={param.max().item():.4f}")

    def forward(self, x):
        return self.net(x)

def train_model(model, train_loader, num_epochs=20, learning_rate=0.001, lambda_penalty=0.1):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    mse_loss = nn.MSELoss()
    
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0
        
        model.train()
        for batch_inputs, batch_targets in train_loader:
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(batch_inputs)
            loss = mse_loss(outputs, batch_targets)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if num_batches % 100 == 0:
                print(f'Epoch {epoch+1} Batch {num_batches} || Loss: {loss.item():.6f}')
        
        avg_loss = total_loss / num_batches
        print(f'Epoch {epoch+1} || Average Loss = {avg_loss:.6f}')

def main(features_path='../../data/processed/features_pca_iv23.csv', 
         iv_path='../../data/processed/predicted_iv23.csv',
         batch_size=128,
         num_epochs=20,
         learning_rate=0.001,
         hidden_size=50):
    
    # Load data
    features_df = pd.read_csv(features_path)
    iv_df = pd.read_csv(iv_path)
    
    dataset = IVDataset(features_df, iv_df)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    input_size = dataset.get_input_size()
    model = IVSDNN(input_size, hidden_size)
    
    print(f"\nModel initialized with {input_size} inputs ({input_size-2} features + m + tau)")
    print(f"Hidden layer size: {hidden_size}")
    
    train_model(model, train_loader, num_epochs, learning_rate)
    
    return model

if __name__ == "__main__":
    model = main()