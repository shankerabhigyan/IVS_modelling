import pandas as pd
import requests
import json
import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
from scipy.stats import norm
from datetime import datetime
import os
from decimal import Decimal, getcontext
from multiprocessing import Pool

import dask.dataframe as dd
from dask.multiprocessing import get
from dask import delayed

from sklearn.linear_model import LinearRegression
# from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler

from api import api_keys
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class getData:
    """
    Use alpha vantage api to get SPY options data
    """
    def __init__(self, symbol='SPY'):
        self.api_keys = api_keys().list
        self.key_iterator = iter(self.api_keys)
        self.symbol = symbol

    def _get_next_key(self):
            try:
                return next(self.key_iterator)
            except StopIteration:
                print("API keys exhausted.")
                return None

    def _fetch_data(self, start_date, end_date):
        query_date_range = pd.date_range(start=start_date, end=end_date)
        data = []
        current_key = self._get_next_key()
        for query_date in query_date_range:
            query_date_str = query_date.strftime('%Y-%m-%d')
            retries = len(self.api_keys) 
            while retries > 0:
                url = f'https://www.alphavantage.co/query?function=HISTORICAL_OPTIONS&symbol={self.symbol}&date={query_date_str}&apikey={current_key}'
                response = requests.get(url)
                d = response.json()
                if 'message' in d and d['message'] == 'success':
                    data.append(d['data'])
                    break  # Successful fetch, move to next date
                elif 'message' in d and d['message']!='success':
                    break # no data for date
                else:
                    print(f"API key {current_key} exhausted: at {query_date_str}")
                    current_key = self._get_next_key()
                    if current_key is None:
                        print(f"API keys exhausted at {query_date_str}")
                        break  # All keys exhausted, stop trying
                    print(f"Using key: {current_key}")
                retries -= 1
        return data

    def run(self, start_date, end_date, savefile):
        data = self._fetch_data(start_date, end_date)
        cwd = os.getcwd()
        savefile = os.path.join(cwd, savefile)
        with open(savefile, 'w') as f:
            json.dump(data, f)
            

class preprocessData:
    """
    create a dataframe from the json file with required features
    """
    def __init__(self, json_path):
        self.json_path = json_path
        getcontext().prec = 8

    def _get_stats(self, obj):
        exp_date = datetime.strptime(obj['expiration'], '%Y-%m-%d')
        curr_date = datetime.strptime(obj['date'], '%Y-%m-%d')
        tte = (exp_date - curr_date).days
        tau = Decimal(tte) / Decimal(365)
        delta = abs(float(obj['delta']))
        if(delta==1):
            delta = 0.999
        elif(delta==0):
            delta = 0.001
        IV = np.float64(obj['implied_volatility'])
        sigma = IV
        q = Decimal('0.01')
        N_inv = norm.ppf(delta * float(np.exp(float(q) * float(tau))))
        sigma = Decimal(sigma)
        m = (Decimal('0.5') * sigma**2 * tau) + (Decimal(N_inv) * sigma * Decimal(tau).sqrt())
        current_date = curr_date.strftime('%Y-%m-%d')
        expiry_date = exp_date.strftime('%Y-%m-%d')
        return [current_date, expiry_date, tte, tau, delta, IV, sigma, N_inv, m]
    
    @delayed
    def delayed_stats(self, obj):
        return self._get_stats(obj)
    
    def fit(self):
        with open(self.json_path, 'r') as f:
            data = json.load(f)
        
        if len(data) == 0:
            logging.error("No data found in the json file")
            return None

        # results = [self.delayed_stats(d) for d in data]
        # results = dd.from_delayed(results)
        # df = results.compute(scheduler='processes')

        # df = dd.from_pandas(df, npartitions=4)
        # merge all sublists in data into one list
        data = [item for sublist in data for item in sublist]
        result = []
        for d in data:
            result.append(self._get_stats(d))
        df = pd.DataFrame(result)
        df.columns = ['date', 'expiry_date', 'tte', 'tau', 'delta', 'IV', 'sigma', 'N_inv', 'm']
        df = df.groupby(['date', 'm', 'tau', 'delta'])['IV'].mean().reset_index()
        df = df[df.IV < 1]
        return df
    
class fitSurface:
    """
    columns = ['date','m','tau','delta','IV']
    fit a surface to the IV data with a discrete grid of m and tau
    following Dumas et al. (1998)
    """
    def __init__(self, df):
        self.df = df
        self.grid_tau = [x/365 for x in [10,30,60,91,122,152,182,273,365,547,730]]
        self.grid_m = [np.log(m) for m in [0.6, 0.8, 0.9, 0.95, 0.975, 1, 1.025, 1.05, 1.1, 1.2, 1.3, 1.5, 1.75, 2]]
        try:
            self.dates = self.df['date'].unique()
        except:
            raise ValueError('Input dataframe must have a "date" column')
        
    def _fit_and_get_model_params(self):
        self.df['m_squared'] = self.df['m']**2
        self.df['tau_squared'] = self.df['tau']**2
        self.df['m_tau'] = self.df['m'] * self.df['tau']
        X = self.df[['m', 'tau', 'm_squared', 'tau_squared', 'm_tau']]
        y = self.df['IV']
        model = LinearRegression()
        model.fit(X, y)
        return model
    
    def _predict(self, tau, m, model):
        m_squared = m**2
        tau_squared = tau**2
        m_tau = m * tau
        input_data = pd.DataFrame({'m': [m], 'tau': [tau], 'm_squared': [m_squared], 'tau_squared': [tau_squared], 'm_tau': [m_tau]})
        iv_ = max(0.01, model.predict(input_data)[0])
        return iv_
    
    def fit(self):
        """Fit surface separately for each date"""
        predicted_iv = pd.DataFrame()
        
        for date in self.dates:
            # Get data for this date only
            date_df = self.df[self.df['date'] == date].copy()
            
            # Fit model for this date
            date_df['m_squared'] = date_df['m']**2
            date_df['tau_squared'] = date_df['tau']**2
            date_df['m_tau'] = date_df['m'] * date_df['tau']
            X = date_df[['m', 'tau', 'm_squared', 'tau_squared', 'm_tau']]
            y = date_df['IV']
            
            model = LinearRegression()
            model.fit(X, y)
            
            # Predict for grid points
            for tau in self.grid_tau:
                for m in self.grid_m:
                    iv = self._predict(tau, m, model)
                    new_row = pd.DataFrame({
                        'date': [date], 
                        'tau': [tau], 
                        'm': [m], 
                        'IV': [iv]
                    })
                    predicted_iv = pd.concat([predicted_iv, new_row])
            
            if len(predicted_iv) % 1000 == 0:  # Progress indicator
                print(f"Processed {len(predicted_iv)} points...")
                        
        predicted_iv.columns = ['date', 'tau', 'm', 'IV']
        return predicted_iv


if __name__ == "__main__":
    # get = getData()
    # get.run('2023-01-01', '2023-01-30', 'spy_options_data_23.json')
    preprocess = preprocessData('../data/raw/yearwise/spy_options_data_22.json')
    df = preprocess.fit()
    print(df.head())
    fit = fitSurface(df)
    predicted_iv = fit.fit()
    print(predicted_iv)
    predicted_iv.to_csv('../data/processed/predicted_iv22.csv', index=False)
    
    
            

    




