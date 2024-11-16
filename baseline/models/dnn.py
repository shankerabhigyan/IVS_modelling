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
    """Calculate calendar spread arbitrage penalty following Section 2.2 equation (2)
    lcal(m,τ) = σ(m,τ) + 2τ∂τσ(m,τ) ≥ 0"""
    
    # Compute partial derivative wrt tau using central difference
    tau_plus = tau + delta_tau
    tau_minus = tau - delta_tau
    sigma = model.predict_iv(features, m, tau)
    sigma_plus = model.predict_iv(features, m, tau_plus)
    sigma_minus = model.predict_iv(features, m, tau_minus)
    
    # Central difference for better accuracy
    d_sigma_dt = (sigma_plus - sigma_minus) / (2 * delta_tau)
    
    # Equation (2) from paper
    l_cal = sigma + 2 * tau * d_sigma_dt
    
    # Return negative part to penalize violations
    return torch.relu(-l_cal).mean()

def butterfly_arbitrage_penalty(model, features, m, tau, delta_m=1e-5):
    """Calculate butterfly arbitrage penalty following Section 2.2 equation (3)
    lbut(m,τ) = (1 - m∂mσ/σ)^2 - (στ∂mσ)^2/4 + τσ∂mmσ ≥ 0"""
    
    # Get central point
    sigma = model.predict_iv(features, m, tau)
    
    # Compute first derivative wrt m using central difference
    m_plus = m + delta_m
    m_minus = m - delta_m
    sigma_plus = model.predict_iv(features, m_plus, tau)
    sigma_minus = model.predict_iv(features, m_minus, tau)
    d_sigma_dm = (sigma_plus - sigma_minus) / (2 * delta_m)
    
    # Compute second derivative wrt m
    d2_sigma_dm2 = (sigma_plus - 2*sigma + sigma_minus) / (delta_m**2)
    
    # Equation (3) from paper
    # Term 1: (1 - m∂mσ/σ)^2
    term1 = (1 - m * d_sigma_dm / (sigma + 1e-8))**2
    
    # Term 2: -(στ∂mσ)^2/4
    term2 = -(sigma * tau * d_sigma_dm)**2 / 4
    
    # Term 3: τσ∂mmσ
    term3 = tau * sigma * d2_sigma_dm2
    
    l_but = term1 + term2 + term3
    
    # Return negative part to penalize violations
    return torch.relu(-l_but).mean()

def large_moneyness_penalty(model, features, m, tau, delta_m=1e-5):
    """Calculate large moneyness penalty following the paper's condition 5:
    'For every τ, σ^2(m,τ) is linear as |m| → +∞'
    This means ∂mm(σ^2) should approach 0 for large |m|"""
    
    # Get central point
    sigma = model.predict_iv(features, m, tau)
    
    # Compute σ^2 at neighboring points
    m_plus = m + delta_m
    m_minus = m - delta_m
    sigma_plus = model.predict_iv(features, m_plus, tau)
    sigma_minus = model.predict_iv(features, m_minus, tau)
    
    # Second derivative of σ^2
    sigma_sq = sigma**2
    sigma_plus_sq = sigma_plus**2
    sigma_minus_sq = sigma_minus**2
    
    d2_sigma_sq_dm2 = (sigma_plus_sq - 2*sigma_sq + sigma_minus_sq) / (delta_m**2)
    
    # Penalty should be higher for larger |m| values
    # Weight the penalty by |m| to focus on asymptotic behavior
    return (torch.abs(d2_sigma_sq_dm2) * torch.abs(m)).mean()

def compute_total_penalty(model, features_batch, m_batch, tau_batch, IC34, IC5):
    """Compute total penalty following Section 3.4 of the paper"""
    
    # Compute penalties on the IC34 grid for calendar spread and butterfly
    cal_penalty = torch.zeros(1, device=features_batch.device)
    but_penalty = torch.zeros(1, device=features_batch.device)
    
    for m, tau in IC34:
        features = features_batch  # Adjust feature tensor for this grid point
        m_tensor = torch.full_like(m_batch, m)
        tau_tensor = torch.full_like(tau_batch, tau)
        
        cal_penalty += calendar_spread_penalty(model, features, m_tensor, tau_tensor)
        but_penalty += butterfly_arbitrage_penalty(model, features, m_tensor, tau_tensor)
    
    cal_penalty /= len(IC34)
    but_penalty /= len(IC34)
    
    # Compute large moneyness penalty on IC5 grid
    large_m_penalty = torch.zeros(1, device=features_batch.device)
    
    for m, tau in IC5:
        features = features_batch  # Adjust feature tensor for this grid point
        m_tensor = torch.full_like(m_batch, m)
        tau_tensor = torch.full_like(tau_batch, tau)
        
        large_m_penalty += large_moneyness_penalty(model, features, m_tensor, tau_tensor)
    
    large_m_penalty /= len(IC5)
    
    return cal_penalty, but_penalty, large_m_penalty 

def create_penalty_grids(mmin=-0.5, mmax=0.5):
    """Create IC34 and IC5 grids following Section 3.4 of the paper"""
    
    # Create T1 grid
    tau_min = 1/365  # Minimum maturity of 1 day
    tau_max = 2      # Maximum maturity of 2 years
    T1 = torch.linspace(tau_min, tau_max, 40)
    
    # Create IC34 grid
    m_min_cube = -((-2*mmin)**(1/3))
    m_max_cube = (2*mmax)**(1/3)
    m_grid_34 = torch.linspace(m_min_cube, m_max_cube, 40)**3
    
    IC34 = [(m, tau) for m in m_grid_34 for tau in T1]
    
    # Create IC5 grid for large moneyness
    m_grid_5 = torch.tensor([6*mmin, 4*mmin, 4*mmax, 6*mmax])
    
    IC5 = [(m, tau) for m in m_grid_5 for tau in T1]
    
    return IC34, IC5

def calculate_mape(targets, outputs):
    epsilon = 1e-10
    # Clip the ratio to prevent extreme values
    percentage_errors = torch.abs((targets - outputs) / (torch.abs(targets) + epsilon)) * 100
    # Optional: clip to reasonable range, e.g., [0, 1000]
    percentage_errors = torch.clip(percentage_errors, 0, 1000)
    return torch.mean(percentage_errors)

def train_model(model, train_loader, num_epochs=20, learning_rate=0.001, lambda_penalty=1.0, wandb=None):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)  # Added weight decay
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    mse_loss = nn.MSELoss()
    IC34, IC5 = create_penalty_grids()
    
    for epoch in range(num_epochs):
        total_loss = 0
        total_penalty = 0
        total_cal_penalty = 0
        total_but_penalty = 0
        total_large_m_penalty = 0
        total_mape = 0
        num_batches = 0
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.to(device)
        
        model.train()
        for batch_inputs, batch_targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            optimizer.zero_grad()
            batch_inputs, batch_targets = batch_inputs.to(device), batch_targets.to(device)
            
            # Forward pass
            outputs = model(batch_inputs)

            mse = mse_loss(outputs, batch_targets)
            mape = calculate_mape(batch_targets, outputs)
            
            # Extract features, m, and tau
            features = batch_inputs[:, :-2]
            m = batch_inputs[:, -2].reshape(-1, 1)
            tau = batch_inputs[:, -1].reshape(-1, 1)
            
            # Calculate penalties with smaller weights initially
            epoch_weight = min(1.0, (epoch + 1) / 5)  # Gradually increase penalty weight
            
            cal_penalty, but_penalty, large_m_penalty = compute_total_penalty(model, features, m, tau, IC34, IC5)
            penalty = cal_penalty + but_penalty + large_m_penalty
            #penalty *= lambda_penalty * epoch_weight
            
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
            total_mape += mape.item()
            num_batches += 1
        
        # Calculate averages
        avg_loss = total_loss / num_batches
        avg_penalty = total_penalty / num_batches
        avg_cal_penalty = total_cal_penalty / num_batches
        avg_but_penalty = total_but_penalty / num_batches
        avg_large_m_penalty = total_large_m_penalty / num_batches
        avg_mape = total_mape / num_batches
        
        # Update learning rate
        scheduler.step(avg_loss)
        
        
        print(f'Epoch {epoch+1} || Loss = {avg_loss:.6f} || '
                f'Penalty = {avg_penalty:.6f} || '
                f'Calendar Penalty = {avg_cal_penalty:.6f} || '
                f'Butterfly Penalty = {avg_but_penalty:.6f} || '
                f'Large Moneyness Penalty = {avg_large_m_penalty:.6f} || '
                f'MAPE = {avg_mape:.6f}')
        wandb.log({
            'epoch': epoch,
            'loss': avg_loss,
            'penalty': avg_penalty,
            'calendar_penalty': avg_cal_penalty,
            'butterfly_penalty': avg_but_penalty,
            'large_moneyness_penalty': avg_large_m_penalty,
            'mape': avg_mape
        })

        # save model
        torch.save(model.state_dict(), 'model.pth')

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
    from tqdm import tqdm
    wandb.init(project="ivs-dnn")
    model = main()