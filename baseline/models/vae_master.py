import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import wandb

class IVSDataset(Dataset):
    """Dataset for implied volatility surfaces where each sample is a full day's surface"""
    def __init__(self, data, device):
        self.data = torch.FloatTensor(data).to(device)
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return self.data[idx]

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, latent_dim=10):
        super(VAE, self).__init__()
        
        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Latent space parameters
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_var = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_var(h)
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var

def loss_function(recon_x, x, mu, log_var, beta=1.0):
    batch_size = x.size(0)
    
    # Reconstruction loss (per dimension)
    recon_loss = F.mse_loss(recon_x, x, reduction='mean')
    
    # KL divergence (normalized by batch size and input dimension)
    kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
    
    # Total loss
    total_loss = recon_loss + beta * kl_loss
    
    mape = torch.mean(torch.abs((x - recon_x) / x))
    
    return total_loss, recon_loss, kl_loss, mape

class IVSFeatureExtractor:
    """Main class for processing IVS data and extracting features using VAE"""
    
    def __init__(self, hidden_dim=128, latent_dim=10, beta=1.0, learning_rate=1e-3):
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.beta = beta
        self.learning_rate = learning_rate
        self.scaler = StandardScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
    def prepare_data(self, df):
        """
        Prepare data by organizing daily IVs into a matrix
        
        Args:
            df: DataFrame with columns [date, tau, m, IV]
        Returns:
            numpy array of shape (n_days, n_points_per_surface)
        """
        # Pivot to get grid points as columns and dates as rows
        grid_points = [(m, t) for m in df['m'].unique() for t in df['tau'].unique()]
        pivoted_data = []
        
        for date in df['date'].unique():
            day_data = df[df['date'] == date]
            day_ivs = []
            for m, t in grid_points:
                iv = day_data[(day_data['m'] == m) & (day_data['tau'] == t)]['IV'].values
                day_ivs.append(iv[0] if len(iv) > 0 else np.nan)
            pivoted_data.append(day_ivs)
            
        pivoted_data = np.array(pivoted_data)
        
        # Scale the data
        scaled_data = self.scaler.fit_transform(pivoted_data)
        return scaled_data
    
    def train(self, train_data, val_df, batch_size=128, n_epochs=200):
        """Train the VAE model"""
        self.wandb = wandb.init(project='ivs-vae')
        self.input_dim = train_data.shape[1]  # Number of IV points per surface
        print(f"Input dimension (IV points per surface): {self.input_dim}")
        
        self.model = VAE(self.input_dim, self.hidden_dim, self.latent_dim)
        self.model.to(self.device)
        
        dataset = IVSDataset(train_data, self.device)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        self.model.train()
        for epoch in range(n_epochs):
            total_epoch_loss = 0
            total_recon_loss = 0
            total_kl_loss = 0
            total_mape = 0
            num_batches = 0
            
            for batch in train_loader:
                optimizer.zero_grad()
                
                recon_batch, mu, log_var = self.model(batch)
                
                loss, recon_loss, kl_loss, mape = loss_function(
                    recon_batch, batch, mu, log_var, self.beta
                )
                
                loss.backward()
                optimizer.step()
                
                total_epoch_loss += loss.item()
                total_recon_loss += recon_loss.item()
                total_kl_loss += kl_loss.item()
                total_mape += mape.item()
                num_batches += 1
            
            avg_loss = total_epoch_loss / num_batches
            avg_recon = total_recon_loss / num_batches
            avg_kl = total_kl_loss / num_batches
            avg_mape = total_mape / num_batches
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1} || Loss = {avg_loss:.4f} || Reconstruction Loss = {avg_recon:.4f} || KL_loss = {avg_kl:.4f} || MAPE = {avg_mape:.4f}")
                
                self.wandb.log({
                    'total_loss': avg_loss,
                    'reconstruction_loss': avg_recon,
                    'kl_loss': avg_kl,
                    'mape': avg_mape
                })

            if (epoch + 1) % 100 == 0:
                print("\nValidating...")
                self.validate(val_df)
                print("\n")
    
    def validate(self, val_df):
        self.model.eval()
        processed_data = self.prepare_data(val_df)
        dataset = IVSDataset(processed_data, self.device)
        loader = DataLoader(dataset, batch_size=len(dataset))
        
        with torch.no_grad():
            batch = next(iter(loader))
            recon_batch, mu, log_var = self.model(batch)
            loss, recon_loss, kl_loss, mape = loss_function(
                recon_batch, batch, mu, log_var, self.beta
            )
            print(f"Validation Loss = {loss:.4f} || Reconstruction Loss = {recon_loss:.4f} || KL_loss = {kl_loss:.4f} || MAPE = {mape:.4f}")
            
            self.wandb.log({
                'val_loss': loss,
                'val_reconstruction_loss': recon_loss,
                'val_kl_loss': kl_loss,
                'val_mape': mape
            })
    
    def extract_features(self, data):
        """
        Extract latent features from data
        
        Args:
            data: numpy array of shape (n_days, n_points_per_surface)
        Returns:
            numpy array of shape (n_days, latent_dim)
        """
        self.model.eval()
        dataset = IVSDataset(data, self.device)
        loader = DataLoader(dataset, batch_size=len(dataset))
        
        with torch.no_grad():
            batch = next(iter(loader))
            mu, _ = self.model.encode(batch)
            
        return mu.cpu().numpy()
    
    def reconstruct_surface(self, features):
        """
        Reconstruct IVS from latent features
        
        Args:
            features: numpy array of shape (n_days, latent_dim)
        Returns:
            numpy array of shape (n_days, n_points_per_surface)
        """
        self.model.eval()
        with torch.no_grad():
            features_tensor = torch.FloatTensor(features).to(self.device)
            reconstructed = self.model.decode(features_tensor)
            
        reconstructed = reconstructed.cpu().numpy()
        return self.scaler.inverse_transform(reconstructed)

if __name__=="__main__":
    # Load data
    df = pd.read_csv('../../data/processed/predicted_iv23.csv')
    val_df = pd.read_csv('../../data/processed/predicted_iv22.csv')

    # Initialize feature extractor
    extractor = IVSFeatureExtractor(
        hidden_dim=4096,
        latent_dim=64,
        beta=1.0,
        learning_rate=0.001
    )

    # Prepare data
    processed_data = extractor.prepare_data(df)
    print(f"Processed data shape: {processed_data.shape}")

    # Train the model
    extractor.train(processed_data, val_df, batch_size=8, n_epochs=25000)

    # Extract features
    features = extractor.extract_features(processed_data)
    df_features = pd.DataFrame(features, columns=[f'feature_{i}' for i in range(features.shape[1])])
    
    # Add date column from original df to features
    df_features['date'] = df['date'].unique()
    
    # Save features
    df_features.to_csv('../../data/processed/features_vae_iv23.csv', index=False)

    # Find min and max for each feature and print
    for i in range(features.shape[1]):
        print(f'Feature {i}: min={features[:,i].min()}, max={features[:,i].max()}')