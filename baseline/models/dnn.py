import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class IVSDNN(nn.Module):
    def __init__(self, input_size, hidden_size=50):
        super(IVSDNN, self).__init__()
        # three hidden layers with 50 neurons each as specified in paper
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

def create_synthetic_grids():
    """synthetic grids to check no-arbitrage conditions"""
    m_min, m_max = np.log(0.6), np.log(2)
    tau_max = 730/365

    # Grid for calendar spread and butterfly arbitrage (IC34)
    m_grid_c34 = np.power(np.linspace(-((-2*m_min)**(1/3)), (2*m_max)**(1/3), 40), 3)
    tau_grid = np.exp(np.linspace(np.log(1/365), np.log(tau_max + 1), 40))
    
    # Grid for large moneyness behavior (IC5)
    m_grid_c5 = np.array([6*m_min, 4*m_min, 4*m_max, 6*m_max])
    
    return m_grid_c34, tau_grid, m_grid_c5

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

class IVSTrainer:
    def __init__(self, model, learning_rate=0.001, lambda_penalty=1.0):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.lambda_penalty = lambda_penalty
        
        # Create synthetic grids for no-arbitrage checks
        self.m_grid_c34, self.tau_grid, self.m_grid_c5 = create_synthetic_grids()
        
    def train_step(self, batch_data):
        """Single training step"""
        self.model.train()
        self.optimizer.zero_grad()
        
        # Unpack batch data
        inputs, targets = batch_data
        
        # Forward pass
        outputs = self.model(inputs)
        
        # MSE loss
        mse_loss = nn.MSELoss()(outputs, targets)
        
        # Calculate no-arbitrage penalties
        cal_penalty = calendar_spread_penalty(self.model, self.m_grid_c34, self.tau_grid)
        but_penalty = butterfly_arbitrage_penalty(self.model, self.m_grid_c34, self.tau_grid)
        large_m_penalty = large_moneyness_penalty(self.model, self.m_grid_c5, self.tau_grid)
        
        # Total loss
        total_loss = mse_loss + self.lambda_penalty * (cal_penalty + but_penalty + large_m_penalty)
        
        # Backward pass and optimization
        total_loss.backward()
        self.optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'mse_loss': mse_loss.item(),
            'calendar_penalty': cal_penalty.item(),
            'butterfly_penalty': but_penalty.item(),
            'large_m_penalty': large_m_penalty.item()
        }

    def train_epoch(self, train_loader):
        """Train for one epoch"""
        epoch_losses = []
        for batch_data in train_loader:
            step_losses = self.train_step(batch_data)
            epoch_losses.append(step_losses)
            
        # Average losses over the epoch
        avg_losses = {k: np.mean([d[k] for d in epoch_losses]) for k in epoch_losses[0].keys()}
        return avg_losses

# Example usage:
def prepare_data(F_t, m, tau):
    """Prepare input data for the DNN"""
    # Concatenate F_t (implied vols at sample points) with m and tau
    inputs = torch.cat([F_t, m.unsqueeze(1), tau.unsqueeze(1)], dim=1)
    return inputs

# Create and train the model
input_size = 156  # 154 (F_t) + 2 (m, tau)
model = IVSDNN(input_size)
trainer = IVSTrainer(model)

# Training loop would look like:
"""
num_epochs = 20
for epoch in range(num_epochs):
    losses = trainer.train_epoch(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"Total Loss: {losses['total_loss']:.6f}")
    print(f"MSE Loss: {losses['mse_loss']:.6f}")
    print(f"Calendar Penalty: {losses['calendar_penalty']:.6f}")
    print(f"Butterfly Penalty: {losses['butterfly_penalty']:.6f}")
    print(f"Large M Penalty: {losses['large_m_penalty']:.6f}")
"""