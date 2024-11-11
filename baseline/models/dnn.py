import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader

class IVDataset(Dataset):
    def __init__(self, df, feature_cols):
        self.data = df          
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
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),  # Add dropout for regularization
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 1),
            nn.Softplus()
        )
        
        # xavier initialisation with smaller bounds
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)
    
    def predict_iv(self, features, m, tau):
        x = torch.cat([features.expand(m.shape[0], -1), m, tau], dim=1)
        return self.net(x)

def safe_divide(a, b, eps=1e-8):
    """Safe division that avoids division by zero"""
    return a / (b + eps)

def calendar_spread_penalty(model, features, m, tau, delta_tau=1e-5):
    """Calculate calendar spread arbitrage penalty with safety checks"""
    with torch.no_grad():
        max_tau = tau.max()
        delta_tau = min(delta_tau, max_tau * 0.1)  # Ensure delta_tau isn't too large
    
    sigma_minus = model.predict_iv(features, m, tau)
    sigma_plus = model.predict_iv(features, m, tau + delta_tau)
    
    d_sigma_dt = (sigma_plus - sigma_minus) / delta_tau
    l_cal = sigma_minus + 2 * tau * d_sigma_dt
    
    # Clip extremely large values
    l_cal = torch.clamp(l_cal, min=-100, max=100)
    return torch.relu(-l_cal).mean()

def butterfly_arbitrage_penalty(model, features, m, tau, delta_m=1e-5):
    """Calculate butterfly arbitrage penalty with safety checks"""
    sigma_mid = model.predict_iv(features, m, tau)
    sigma_plus = model.predict_iv(features, m + delta_m, tau)
    sigma_minus = model.predict_iv(features, m - delta_m, tau)
    
    d_sigma_dm = (sigma_plus - sigma_minus) / (2 * delta_m)
    d2_sigma_dm2 = (sigma_plus - 2*sigma_mid + sigma_minus) / (delta_m**2)
    
    term1 = (1 - m * safe_divide(d_sigma_dm, sigma_mid))**2
    term2 = -(tau * sigma_mid * d_sigma_dm)**2 / 4
    term3 = tau * sigma_mid * d2_sigma_dm2
    
    l_but = term1 + term2 + term3
    # Clip extremely large values
    l_but = torch.clamp(l_but, min=-100, max=100)
    return torch.relu(-l_but).mean()

def large_moneyness_penalty(model, features, m, tau, delta_m=1e-5):
    """Calculate large moneyness penalty with safety checks"""
    sigma_mid = model.predict_iv(features, m, tau)
    sigma_plus = model.predict_iv(features, m + delta_m, tau)
    sigma_minus = model.predict_iv(features, m - delta_m, tau)
    
    d_sigma_dm = (sigma_plus - sigma_minus) / (2 * delta_m)
    d2_sigma_dm2 = (sigma_plus - 2*sigma_mid + sigma_minus) / (delta_m**2)
    
    penalty = torch.abs(sigma_mid * d2_sigma_dm2 + d_sigma_dm**2)
    # Clip extremely large values
    penalty = torch.clamp(penalty, min=0, max=100)
    return penalty.mean()

def train_model(model, train_loader, num_epochs=20, learning_rate=0.001, lambda_penalty=1.0, wandb=None):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)  # Added weight decay
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    mse_loss = nn.MSELoss()
    
    for epoch in range(num_epochs):
        total_loss = 0
        total_penalty = 0
        total_cal_penalty = 0
        total_but_penalty = 0
        total_large_m_penalty = 0
        num_batches = 0
        
        model.train()
        for batch_inputs, batch_targets in train_loader:
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(batch_inputs)

            mse = mse_loss(outputs, batch_targets)
            
            # Extract features, m, and tau
            features = batch_inputs[:, :-2]
            m = batch_inputs[:, -2].reshape(-1, 1)
            tau = batch_inputs[:, -1].reshape(-1, 1)
            
            # Calculate penalties with smaller weights initially
            epoch_weight = min(1.0, (epoch + 1) / 5)  # Gradually increase penalty weight
            cal_penalty = calendar_spread_penalty(model, features, m, tau)
            but_penalty = butterfly_arbitrage_penalty(model, features, m, tau)
            large_m_penalty = large_moneyness_penalty(model, features, m, tau)
            
            penalty = lambda_penalty * epoch_weight * (cal_penalty + but_penalty + large_m_penalty)
            loss = mse + penalty
            
            # Backward pass with gradient clipping
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Accumulate metrics
            total_loss += loss.item()
            total_penalty += penalty.item()
            total_cal_penalty += cal_penalty.item()
            total_but_penalty += but_penalty.item()
            total_large_m_penalty += large_m_penalty.item()
            num_batches += 1
        
        # Calculate averages
        avg_loss = total_loss / num_batches
        avg_penalty = total_penalty / num_batches
        avg_cal_penalty = total_cal_penalty / num_batches
        avg_but_penalty = total_but_penalty / num_batches
        avg_large_m_penalty = total_large_m_penalty / num_batches
        
        # Update learning rate
        scheduler.step(avg_loss)
        
        
        print(f'Epoch {epoch+1} || Loss = {avg_loss:.6f} || '
                f'Penalty = {avg_penalty:.6f} || '
                f'Calendar Penalty = {avg_cal_penalty:.6f} || '
                f'Butterfly Penalty = {avg_but_penalty:.6f} || '
                f'Large Moneyness Penalty = {avg_large_m_penalty:.6f}')
        wandb.log({
            'epoch': epoch,
            'loss': avg_loss,
            'penalty': avg_penalty,
            'calendar_penalty': avg_cal_penalty,
            'butterfly_penalty': avg_but_penalty,
            'large_moneyness_penalty': avg_large_m_penalty
        })

def main(features_path='../../data/processed/features_pca_iv23.csv', 
         iv_path='../../data/processed/predicted_iv23.csv',
         batch_size=512,
         num_epochs=100,
         learning_rate=0.001,
         hidden_size=256,
         lambda_penalty=1):  
    
    # Load and prepare data
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
    import wandb
    wandb.init(project="ivs-dnn")
    model = main()