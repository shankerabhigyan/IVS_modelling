import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wandb

class CustomLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(CustomLSTMCell, self).__init__()
        self.hidden_dim = hidden_dim
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
        self.W_i = nn.Linear(input_dim + hidden_dim, hidden_dim).to(self.device)
        self.W_f = nn.Linear(input_dim + hidden_dim, hidden_dim).to(self.device)
        self.W_o = nn.Linear(input_dim + hidden_dim, hidden_dim).to(self.device)
        self.W_g = nn.Linear(input_dim + hidden_dim, hidden_dim).to(self.device)
        
    def forward(self, Z, h_prev, c_prev):
        Z = Z.to(self.device)
        h_prev = h_prev.to(self.device)
        c_prev = c_prev.to(self.device)
        combined = torch.cat((Z, h_prev), dim=1)
        
        i = torch.sigmoid(self.W_i(combined))
        f = torch.sigmoid(self.W_f(combined))
        o = torch.sigmoid(self.W_o(combined))
        g = torch.tanh(self.W_g(combined))
        
        c = f * c_prev + i * g
        h = o * torch.tanh(c)
        
        return h, c
    
class CustomLSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(CustomLSTMModel, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.hidden_dim = hidden_dim
        self.lstm_cell = CustomLSTMCell(input_dim, hidden_dim).to(self.device)
        self.dense = nn.Linear(hidden_dim, output_dim).to(self.device)
        
    def forward(self, inputs):
        inputs = inputs.unsqueeze(1)
        batch_size, seq_len, _ = inputs.size()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        h, c = torch.zeros(batch_size, self.hidden_dim, device=device), torch.zeros(batch_size, self.hidden_dim, device=device)
        for t in range(seq_len):
            Z_t = inputs[:, t, :]
            h, c = self.lstm_cell(Z_t, h, c)    
        output = self.dense(h)
        return output
    
    def load_model(self, model_path):
        self.load_state_dict(torch.load(model_path))
        print(f'Model loaded from {model_path}')

    def predict(self, inputs):
        self.eval()
        with torch.no_grad():
            inputs = inputs.unsqueeze(0)
            output = self(inputs)
        return output
    
class ModelManager:
    def __init__(self, input_dim, hidden_dim, output_dim, learning_rate=0.001, model_path='model.pth'):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = CustomLSTMModel(input_dim, hidden_dim, output_dim).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.model_path = model_path
        self.wandb = wandb.init(project='IVS_LSTM')

    def train(self, train_data, train_targets, batch_size=128, epochs=10, val_data=None, val_labels=None):
        train_dataset = TensorDataset(train_data, train_targets)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Prepare validation data if provided
        if val_data is not None and val_labels is not None:
            val_dataset = TensorDataset(val_data, val_labels)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            validate = True
        else:
            validate = False

        self.model.train()  # Set the model to training mode
        for epoch in range(epochs):
            running_loss = 0.0
            running_mape = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()  # Zero the parameter gradients
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                mape = torch.mean(torch.abs((labels - outputs) / labels)) * 100
                loss.backward()  # Backpropagation
                self.optimizer.step()  # Optimize
                running_loss += loss.item()
                running_mape += mape.item()

            print(f'Epoch {epoch+1} Loss: {running_loss / len(train_loader)} MAPE: {running_mape / len(train_loader)}')
            self.wandb.log({
                'train_loss': running_loss / len(train_loader),
                'train_mape': running_mape / len(train_loader)
            })
            if validate:
                self.validate(val_loader)

        print("Training complete.")
        self.save_model()

    def validate(self, val_loader):
        self.model.eval()  # Set the model to evaluation mode
        total_loss = 0.0
        with torch.no_grad():  # No gradients needed for validation
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs).to(self.device)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
        average_loss = total_loss / len(val_loader)
        print(f'Validation Loss: {average_loss}')
        self.model.train()  # Set back to training mode

    def save_model(self):
        torch.save(self.model.state_dict(), self.model_path)
        print(f'Model saved to {self.model_path}')

class DatasetManager:
    def __init__(self, csv_path, ma_list=[1,5,22]):
        self.df = pd.read_csv(csv_path)
        self.ma_list = ma_list
    
    def make_train_target_pairs(self, feature_cols=None):
        # dataset is sorted by date
        # we pick a day's feature as target and use MA of previous one, five and twenty-two days as features
        # we create a torch dataset with these pairs
        # feature1 , feature2, feature3 are the columns indexed by date
        features = []
        targets = []
        if feature_cols is None:
            for i in range(22, len(self.df)):
                # target tensor (feature1, feature2, feature3)
                target = torch.tensor(self.df.iloc[i][['feature1', 'feature2', 'feature3']].astype(float).values, dtype=torch.float32)
                targets.append(target)
                # feature tensor (ma1, ma5, ma22)
                ma1 = torch.tensor(self.df.iloc[i-1][['feature1', 'feature2', 'feature3']].astype(float).values, dtype=torch.float32)
                ma5 = torch.tensor(self.df.iloc[i-5:i][['feature1', 'feature2', 'feature3']].mean().astype(float).values, dtype=torch.float32)
                ma22 = torch.tensor(self.df.iloc[i-22:i][['feature1', 'feature2', 'feature3']].mean().astype(float).values, dtype=torch.float32)
                feature = torch.cat((ma1, ma5, ma22))
                features.append(feature)
            return torch.stack(features), torch.stack(targets)
        
        else:
            for i in range(22, len(self.df)):
                # target tensor (feature1, feature2, feature3)
                target = torch.tensor(self.df.iloc[i][feature_cols].astype(float).values, dtype=torch.float32)
                targets.append(target)
                # feature tensor (ma1, ma5, ma22)
                ma1 = torch.tensor(self.df.iloc[i-1][feature_cols].astype(float).values, dtype=torch.float32)
                ma5 = torch.tensor(self.df.iloc[i-5:i][feature_cols].mean().astype(float).values, dtype=torch.float32)
                ma22 = torch.tensor(self.df.iloc[i-22:i][feature_cols].mean().astype(float).values, dtype=torch.float32)
                feature = torch.cat((ma1, ma5, ma22))
                features.append(feature)
            return torch.stack(features), torch.stack(targets)

       
if __name__ == '__main__':
    print("Testing Script...")
    # Load the dataset
    dataset = DatasetManager('../../data/processed/features_pca_iv23.csv')
    features, targets = dataset.make_train_target_pairs()
    print('Features shape:', features.shape)
    print('Targets shape:', targets.shape)

    # Split the dataset into training and validation sets
    # split = int(0.8 * len(features))
    # train_features, val_features = features[:split], features[split:]
    # train_targets, val_targets = targets[:split], targets[split:]

    # no split
    train_features, train_targets = features, targets

    # Initialize the model
    print('Initializing model...')
    model_path = './experiments/test_lstm1.pth'
    model = ModelManager(input_dim=9, hidden_dim=512, output_dim=3, model_path=model_path)
    model.train(train_features, train_targets, epochs=600)

    #val_loader = DataLoader(TensorDataset(val_features, val_targets), batch_size=1, shuffle=False)
    #model.validate(val_loader)

    model.save_model()