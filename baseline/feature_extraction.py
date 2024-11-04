import numpy as np
import dask.dataframe as dd
import pandas as pd

class featureExtractor:
    """
    extract features from the tabular data for our LSTM model
    We will be using dask for parallel processing for handling large datasets
    """
    def __init__(self, df_path, save_path):
        self.save_path = save_path
        if type(df_path)==list:
            self.df = pd.concat([pd.read_csv(path) for path in df_path])
        else:
            self.df = pd.read_csv(df_path)

    def transform(self):
        self.df.sort_values(by=['date', 'tau', 'm'], inplace=True)
        self.df.reset_index(drop=True, inplace=True)
        self.df['Xt'] = np.log(self.df['IV'])
        # calculate Xt- X0
        self.df['U_mt'] = self.df.groupby(['date'])['Xt'].transform(lambda x: x - x.iloc[0])
        self.df.replace(np.nan, 0, inplace=True)

    def fit(self):
        dates = self.df['date'].unique()
        starting_date = dates[0]
        pivot = self.df.pivot(index='date',columns=['m','tau'],values='U_mt')
        K = pivot.cov().values
        eigenvalues, eigenvectors = np.linalg.eigh(K)
        sorted_indices = np.argsort(eigenvalues)[::-1]
        sorted_eigenvecs = eigenvectors[:, sorted_indices]
        k=3
        top_eigenvecs = sorted_eigenvecs[:, :k]
        features = pd.DataFrame()
        for date in dates[1:]:
            u = self.df[self.df['date'] == date].sort_values(by=['m', 'tau'])['U_mt'].values
            feature = np.dot(u, top_eigenvecs)
            features[date] = feature
        features = features.T
        features['date'] = dates[1:]
        features = features[['date', 0, 1, 2]]
        features.columns = ['date', 'feature1', 'feature2', 'feature3']
        features.reset_index(drop=True, inplace=True)
        features.to_csv(self.save_path, index=False)
        print('Features saved to', self.save_path)
        return features
    
if __name__=="__main__":
    # path = "data/predicted_iv_jan24.csv"
    paths = ["../data/processed/predicted_iv23.csv"]
    save_path = "../data/processed/features_pca_iv23.csv"
    #paths = ["../data/old/predicted_iv_jan24.csv"]
    #save_path = "../data/features_q1_24.csv"
    fe = featureExtractor(paths, save_path)
    fe.transform()
    fe.fit()
