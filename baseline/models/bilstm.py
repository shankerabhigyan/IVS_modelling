import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wandb

class CustomBiLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(CustomBiLSTMCell, self).__init__()
        self.hidden_dim = hidden_dim
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Forward direction
        self.W_i_forward = nn.Linear(input_dim + hidden_dim, hidden_dim).to(self.device)
        self.W_f_forward = nn.Linear(input_dim + hidden_dim, hidden_dim).to(self.device)
        self.W_o_forward = nn.Linear(input_dim + hidden_dim, hidden_dim).to(self.device)
        self.W_g_forward = nn.Linear(input_dim + hidden_dim, hidden_dim).to(self.device)
        
        # Backward direction
        self.W_i_backward = nn.Linear(input_dim + hidden_dim, hidden_dim).to(self.device)
        self.W_f_backward = nn.Linear(input_dim + hidden_dim, hidden_dim).to(self.device)
        self.W_o_backward = nn.Linear(input_dim + hidden_dim, hidden_dim).to(self.device)
        self.W_g_backward = nn.Linear(input_dim + hidden_dim, hidden_dim).to(self.device)
        
    def forward(self, Z, h_prev, c_prev, direction='forward'):
        Z = Z.to(self.device)
        h_prev = h_prev.to(self.device)
        c_prev = c_prev.to(self.device)
        combined = torch.cat((Z, h_prev), dim=1)
        
        if direction == 'forward':
            i = torch.sigmoid(self.W_i_forward(combined))
            f = torch.sigmoid(self.W_f_forward(combined))
            o = torch.sigmoid(self.W_o_forward(combined))
            g = torch.tanh(self.W_g_forward(combined))
        else:
            i = torch.sigmoid(self.W_i_backward(combined))
            f = torch.sigmoid(self.W_f_backward(combined))
            o = torch.sigmoid(self.W_o_backward(combined))
            g = torch.tanh(self.W_g_backward(combined))
        
        c = f * c_prev + i * g
        h = o * torch.tanh(c)
        
        return h, c

class CustomBiLSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(CustomBiLSTMModel, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.hidden_dim = hidden_dim
        self.lstm_cell = CustomBiLSTMCell(input_dim, hidden_dim).to(self.device)
        # Output layer takes concatenated hidden states from both directions
        self.dense = nn.Linear(hidden_dim * 2, output_dim).to(self.device)
        
    def forward(self, inputs):
        inputs = inputs.unsqueeze(1)
        batch_size, seq_len, _ = inputs.size()
        h_forward = torch.zeros(batch_size, self.hidden_dim, device=self.device)
        c_forward = torch.zeros(batch_size, self.hidden_dim, device=self.device)
        h_backward = torch.zeros(batch_size, self.hidden_dim, device=self.device)
        c_backward = torch.zeros(batch_size, self.hidden_dim, device=self.device)

        # Forward pass
        forward_states = []
        for t in range(seq_len):
            Z_t = inputs[:, t, :]
            h_forward, c_forward = self.lstm_cell(Z_t, h_forward, c_forward, 'forward')
            forward_states.append(h_forward)

        # Backward pass
        backward_states = []
        for t in range(seq_len-1, -1, -1):
            Z_t = inputs[:, t, :]
            h_backward, c_backward = self.lstm_cell(Z_t, h_backward, c_backward, 'backward')
            backward_states.insert(0, h_backward)

        # Concatenate forward and backward states
        combined_states = torch.cat((forward_states[-1], backward_states[-1]), dim=1)
        output = self.dense(combined_states)
        return output

class BiLSTMModelManager:
    def __init__(self, input_dim, hidden_dim, output_dim, learning_rate=0.001, model_path='bilstm_model.pth'):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = CustomBiLSTMModel(input_dim, hidden_dim, output_dim).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.model_path = model_path
        self.wandb = wandb.init(project='IVS_LSTM')

    def train(self, train_data, train_targets, batch_size=128, epochs=10, val_data=None, val_labels=None):
        train_dataset = TensorDataset(train_data, train_targets)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        if val_data is not None and val_labels is not None:
            val_dataset = TensorDataset(val_data, val_labels)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            validate = True
        else:
            validate = False

        self.model.train()
        for epoch in range(epochs):
            running_loss = 0.0
            running_mape = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                mape = torch.mean(torch.abs((labels - outputs) / labels)) * 100
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                running_mape += mape.item()

            avg_loss = running_loss / len(train_loader)
            avg_mape = running_mape / len(train_loader)
            print(f'Epoch {epoch+1} Loss: {avg_loss:.6f} MAPE: {avg_mape:.2f}%')
            
            self.wandb.log({
                'train_loss': avg_loss,
                'train_mape': avg_mape
            })
            
            if validate:
                val_loss = self.validate(val_loader)
                self.wandb.log({'val_loss': val_loss})

        print("Training complete.")
        self.save_model()

    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
        
        avg_loss = total_loss / len(val_loader)
        print(f'Validation Loss: {avg_loss:.6f}')
        self.model.train()
        return avg_loss

    def save_model(self):
        torch.save(self.model.state_dict(), self.model_path)
        print(f'Model saved to {self.model_path}')

class BiLSTMDatasetManager:
    def __init__(self, csv_path, ma_list=[1,5,22]):
        self.df = pd.read_csv(csv_path)
        self.ma_list = ma_list
    
    def make_train_target_pairs(self):
        features = []
        targets = []
        for i in range(22, len(self.df)):
            target = torch.tensor(self.df.iloc[i][['feature1', 'feature2', 'feature3']].astype(float).values, dtype=torch.float32)
            targets.append(target)
            
            ma1 = torch.tensor(self.df.iloc[i-1][['feature1', 'feature2', 'feature3']].astype(float).values, dtype=torch.float32)
            ma5 = torch.tensor(self.df.iloc[i-5:i][['feature1', 'feature2', 'feature3']].mean().astype(float).values, dtype=torch.float32)
            ma22 = torch.tensor(self.df.iloc[i-22:i][['feature1', 'feature2', 'feature3']].mean().astype(float).values, dtype=torch.float32)
            feature = torch.cat((ma1, ma5, ma22))
            features.append(feature)
            
        return torch.stack(features), torch.stack(targets)

if __name__ == '__main__':
    print("Testing BiLSTM Script...")
    # Load the dataset
    dataset = BiLSTMDatasetManager('../../data/processed/features_pca_iv23.csv')
    features, targets = dataset.make_train_target_pairs()
    print('Features shape:', features.shape)
    print('Targets shape:', targets.shape)

    # Initialize and train the model
    print('Initializing BiLSTM model...')
    model_path = './experiments/test_bilstm1.pth'
    model = BiLSTMModelManager(input_dim=9, hidden_dim=512, output_dim=3, model_path=model_path)
    model.train(features, targets, epochs=600)