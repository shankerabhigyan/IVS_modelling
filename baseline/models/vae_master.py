import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

import wandb

# class IVSDataset(Dataset):
#     """Custom dataset for implied volatility surface data"""
#     def __init__(self, data):
#         # Each data point should be a snapshot of the surface
#         # Shape: (n_samples, grid_size)
#         self.data = torch.FloatTensor(data)
    
#     def __len__(self):
#         return len(self.data)
    
#     def __getitem__(self, idx):
#         return self.data[idx]

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
    """
    VAE loss function = reconstruction loss + KL divergence
    
    Args:
        recon_x: reconstructed input
        x: original input
        mu: mean of the latent distribution
        log_var: log variance of the latent distribution
        beta: weight of the KL divergence term (beta-VAE parameter)
    
    Returns:
        total_loss: weighted sum of reconstruction loss and KL divergence
        recon_loss: reconstruction loss component
        kl_loss: KL divergence component
    """
    batch_size = x.size(0)
    
    # Reconstruction loss (per dimension)
    recon_loss = F.mse_loss(recon_x, x, reduction='mean')
    
    # KL divergence (normalized by batch size and input dimension)
    kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
    
    # Total loss
    total_loss = recon_loss + beta * kl_loss

    mape = torch.mean(torch.abs((x - recon_x) / x))
    
    return total_loss, recon_loss, kl_loss, mape

class IVSDataset(Dataset):
    def __init__(self, data, device):
        self.data = torch.FloatTensor(data).to(device)
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return self.data[idx]

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
        Prepare data by pivoting and standardizing
        
        Args:
            df: DataFrame with columns [date, tau, m, IV]
        Returns:
            numpy array of shape (n_dates, n_points_per_surface)
        """
        pivot_df = df.pivot(index='date', columns=['tau', 'm'], values='IV')
        
        scaled_data = self.scaler.fit_transform(pivot_df)
        # print(f"sample scaled data: {scaled_data[0]}")
        # print(f"scaled data shape: {scaled_data.shape}")
        return scaled_data
    
    def train(self, train_data, val_df, batch_size=128, n_epochs=200):
        """Train the VAE model"""
        self.wandb = wandb.init(project='ivs-vae')
        self.input_dim = train_data.shape[1]
        self.model = VAE(self.input_dim, self.hidden_dim, self.latent_dim)
        self.model.to(self.device)
        
        dataset = IVSDataset(train_data, self.device)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        avg_100_loss = 0
        
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
            data: numpy array of shape (n_samples, n_points_per_surface)
        Returns:
            numpy array of shape (n_samples, latent_dim)
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
            features: numpy array of shape (n_samples, latent_dim)
        Returns:
            numpy array of shape (n_samples, n_points_per_surface)
        """
        self.model.eval()
        with torch.no_grad():
            features_tensor = torch.FloatTensor(features).to(self.device)
            reconstructed = self.model.decode(features_tensor)
            
        # Move back to CPU and convert to numpy for inverse transform
        reconstructed = reconstructed.cpu().numpy()
        # Inverse transform to get back to original scale
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

    # Train the model
    extractor.train(processed_data, val_df, batch_size=8, n_epochs=25000)

    # Extract features
    features = extractor.extract_features(processed_data)
    # custom = CustomScaler()
    # features = custom.fit_transform(features)
    df_features = pd.DataFrame(features, columns=[f'feature_{i}' for i in range(features.shape[1])])
    # df_features = pd.DataFrame(features, columns=[f'feature_{i}' for i in range(features.shape[1])])
    df = pd.concat([df, df_features], axis=1)
    print(df.head())
    df.to_csv('../../data/processed/features_vae_iv23.csv', index=False)

    # find min and max for each feature and print
    for i in range(features.shape[1]):
        print(f'Feature {i}: min={features[:,i].min()}, max={features[:,i].max()}')


