import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader

class IVDataset(Dataset):
    def __init__(self, features_df, iv_df):
        # Merge features with IV data on date
        self.data = iv_df.merge(features_df, on='date', how='left')
        
        # Identify feature columns (all columns except 'date')
        self.feature_cols = [col for col in features_df.columns if col != 'date']
        
        # Convert to tensors
        self.features = torch.tensor(self.data[self.feature_cols].values, dtype=torch.float32)
        self.m = torch.tensor(self.data['m'].values, dtype=torch.float32).reshape(-1, 1)
        self.tau = torch.tensor(self.data['tau'].values, dtype=torch.float32).reshape(-1, 1)
        self.iv = torch.tensor(self.data['IV'].values, dtype=torch.float32).reshape(-1, 1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Return features, m, tau as input and IV as target
        input_tensor = torch.cat([self.features[idx], self.m[idx], self.tau[idx]], dim=0)
        return input_tensor, self.iv[idx]
    
    def get_input_size(self):
        # Return total input size (num_features + m + tau)
        return len(self.feature_cols) + 2

class IVSDNN(nn.Module):
    def __init__(self, input_size, hidden_size=50):
        super(IVSDNN, self).__init__()
        # Input size will be (num_features + m + tau)
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Softplus()  # ensures output is positive and twice differentiable
        )
        
        # xavier initialisation
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)

def calendar_spread_penalty(sigma, m, tau, delta_tau=1e-5):
    """Calculate calendar spread arbitrage penalty"""
    # Approximate time derivative using finite difference
    sigma_plus = sigma(m, tau + delta_tau)
    sigma_minus = sigma(m, tau)
    d_sigma_dt = (sigma_plus - sigma_minus) / delta_tau
    
    l_cal = sigma_minus + 2 * tau * d_sigma_dt
    return torch.relu(-l_cal).mean()

def butterfly_arbitrage_penalty(sigma, m, tau, delta_m=1e-5):
    """Calculate butterfly arbitrage penalty"""
    # Approximate first and second derivatives w.r.t. moneyness
    sigma_plus = sigma(m + delta_m, tau)
    sigma_minus = sigma(m - delta_m, tau)
    sigma_mid = sigma(m, tau)
    
    d_sigma_dm = (sigma_plus - sigma_minus) / (2 * delta_m)
    d2_sigma_dm2 = (sigma_plus - 2*sigma_mid + sigma_minus) / (delta_m**2)
    
    term1 = (1 - m*d_sigma_dm/sigma_mid)**2
    term2 = -(tau * sigma_mid * d_sigma_dm)**2 / 4
    term3 = tau * sigma_mid * d2_sigma_dm2
    
    l_but = term1 + term2 + term3
    return torch.relu(-l_but).mean()

def large_moneyness_penalty(sigma, m, tau):
    """Calculate penalty for large moneyness behavior"""
    delta_m = 1e-5
    sigma_plus = sigma(m + delta_m, tau)
    sigma_minus = sigma(m - delta_m, tau)
    sigma_mid = sigma(m, tau)
    
    d_sigma_dm = (sigma_plus - sigma_minus) / (2 * delta_m)
    d2_sigma_dm2 = (sigma_plus - 2*sigma_mid + sigma_minus) / (delta_m**2)
    
    return torch.abs(sigma_mid * d2_sigma_dm2 + d_sigma_dm**2).mean()

def train_model(model, train_loader, num_epochs=20, learning_rate=0.001, lambda_penalty=1.0):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    mse_loss = nn.MSELoss()
    
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_inputs, batch_targets in train_loader:
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(batch_inputs)
            
            # MSE loss
            loss = mse_loss(outputs, batch_targets)
            
            # no-arbitrage penalties
            # m and tau are always the last two columns
            m = batch_inputs[:, -2].reshape(-1, 1)  # second-to-last column
            tau = batch_inputs[:, -1].reshape(-1, 1)  # last column
            
            cal_penalty = calendar_spread_penalty(model, m, tau)
            but_penalty = butterfly_arbitrage_penalty(model, m, tau)
            large_m_penalty = large_moneyness_penalty(model, m, tau)
            
            total_penalty = lambda_penalty * (cal_penalty + but_penalty + large_m_penalty)
            loss += total_penalty
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.6f}')

def main(features_path='data/processed/features_pca_iv23.csv', 
         iv_path='data/processed/predicted_iv23.csv',
         batch_size=1024,
         num_epochs=20,
         learning_rate=0.001,
         hidden_size=50,
         lambda_penalty=1.0):
    """
    Main function with configurable parameters
    
    Args:
        features_path (str): Path to features CSV file
        iv_path (str): Path to implied volatility CSV file
        batch_size (int): Batch size for training
        num_epochs (int): Number of training epochs
        learning_rate (float): Learning rate for optimizer
        hidden_size (int): Number of neurons in hidden layers
        lambda_penalty (float): Weight for no-arbitrage penalties
    """
    features_df = pd.read_csv(features_path)
    iv_df = pd.read_csv(iv_path)
    
    dataset = IVDataset(features_df, iv_df)
    
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    input_size = dataset.get_input_size()
    model = IVSDNN(input_size, hidden_size)
    print(f"Model initialized with {input_size} inputs ({input_size-2} features + m + tau)")
    print(f"Hidden layer size: {hidden_size}")
    train_model(model, train_loader, num_epochs, learning_rate, lambda_penalty)
    return model

if __name__ == "__main__":
    model = main()
    
    """
    model = main(
        features_path='path/to/features.csv',
        iv_path='path/to/iv.csv',
        batch_size=512,
        num_epochs=30,
        learning_rate=0.0005,
        hidden_size=100,
        lambda_penalty=0.5
    )
    """