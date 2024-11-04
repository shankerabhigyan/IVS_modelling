from models.lstm import CustomLSTMCell, CustomLSTMModel
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

class trainLSTM:
    def __init__(self, df_path):
        self.df = pd.read_csv(df_path)
        self.model = CustomLSTMModel(3, 100, 1) # input_dim, hidden_dim, output_dim
        columns = self.df.columns
        print(columns)

    def transform(self):
        k = self.df.shape[1]
        # feature vector
        train_df = pd.DataFrame()
        features = train_df[['feature1', 'feature2', 'feature3']]
        for i in range(len(features)-22):
            k = i+21
            ma22 = features[k-21:k].mean()
            ma5 = features[k-5:k].mean()
            prev = features[k-1]
            feature = pd.concat([ma22, ma5, prev])
            target = features[k+1]
            row = pd.concat([feature, target])
            train_df = pd.concat([train_df, row])
        return train_df

    def fit(self):
        train_df = self.transform()
        print(train_df.head())
        for i in range(100):
            for row in data:
                print(row)
                features = row[:-1].values
                target = row[-1]
                features = torch.tensor(features, dtype=torch.float32).reshape(1, 3)
                target = torch.tensor(target, dtype=torch.float32).reshape(1, 1)
                optimizer = optim.Adam(self.model.parameters(), lr=0.001)
                criterion = nn.MSELoss()
                optimizer.zero_grad()
                output = self.model(features)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                print(f"Epoch {i}, Loss: {loss.item()}")
        return self.model
    
if __name__=="__main__":
    df_path = "data/features_janfeb.csv"
    train = trainLSTM(df_path)
    model = train.fit()
    torch.save(model.state_dict(), 'lstm.pth')
