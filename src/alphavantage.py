import pandas as pd
import requests
import json
import numpy as np
from scipy.stats import norm
from datetime import datetime
import os
from decimal import Decimal, getcontext
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class api_keys:
    def __init__(self):
        self.session = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(self.session)
        self.list = (
           # "LULKG4NIIZEJI6D2",
            "YPOLRZMHBE5X9RKF",
            "0C4JF7CVP6E14A6N",
            "2YOP96J6F1HGCWPA", # d2
            "3ZFGTFL238KPNCIY", # d3
            "BXPI10OBKLNR0QQ8", # d4
            "K8Q66F1AA7SDI317", # d5
            "MC1DGRM5VJ6QQ4U3", # mp1
            "KTPJ50N8JC8RKEGK", # mp2
            "BC3IFP07ZZFZICJB", # mp3
            "6Y3LB06GFNIU0OB1", # mp4
            "60B1KPJJNGHZIDTC", # bits 1
            "F0NZ9L4D8K4V3QAV", # bits 2
            "8TV5GX40IW442BWY", # bits 3
            "JDGY7JGQ2CGEZ6ER", # d6
            "JT9NDBWC8VFGML3R", # d7
            "SOYVTYHXUFZNH2A1", # d8
            "G5MW2DYRW52UQKEO", # d9
            "QBYT5P0FZ79JJBHT", # d10
            "BY7HI8VLA18UUNVY", #d11
            "WUD5HQDYD0KDK2MH" #d12
        )

    def iterator(self):
        for i in self.list:
            yield i

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
            logging.warning("API keys exhausted.")
            return None

    def _fetch_data(self, start_date, end_date):
        query_date_range = pd.date_range(start=start_date, end=end_date)
        data = []
        current_key = self._get_next_key()
        
        for query_date in query_date_range:
            query_date_str = query_date.strftime('%Y-%m-%d')
            retries = len(self.api_keys)
            
            while retries > 0 and current_key is not None:
                url = f'https://www.alphavantage.co/query?function=HISTORICAL_OPTIONS&symbol={self.symbol}&date={query_date_str}&apikey={current_key}'
                try:
                    logging.info(f"Trying to fetch data for {query_date_str} with key {current_key[:4]}...")
                    response = requests.get(url, timeout=10)
                    
                    # Check if response status is OK
                    if response.status_code != 200:
                        logging.warning(f"Error: Received status code {response.status_code} for {query_date_str}")
                        if response.status_code == 429:  # Too Many Requests
                            logging.warning("Rate limit hit, trying next key...")
                            current_key = self._get_next_key()
                            continue
                        # Other error, skip this date
                        break
                    
                    # Check if response is empty
                    if not response.text or response.text.strip() == '':
                        logging.warning(f"Empty response received for {query_date_str}, trying next key...")
                        current_key = self._get_next_key()
                        continue
                    
                    try:
                        # Parse JSON response
                        d = response.json()
                        
                        if 'message' in d and d['message'] == 'success':
                            data.append(d['data'])
                            logging.info(f"Successfully fetched data for {query_date_str}")
                            break  # Successful fetch, move to next date
                        elif 'message' in d and d['message'] != 'success':
                            logging.info(f"No data found for {query_date_str}")
                            logging.info(d)
                            break  # No data for date, move to next date
                        else:
                            logging.warning(f"API key {current_key[:4]}... exhausted at {query_date_str}")
                            current_key = self._get_next_key()
                            if current_key is None:
                                logging.error(f"All API keys exhausted at {query_date_str}")
                                break
                            logging.info(f"Using key: {current_key[:4]}...")
                    
                    except json.JSONDecodeError as e:
                        logging.error(f"JSON decode error for {query_date_str}: {e}")
                        logging.error(f"Response text: {response.text[:100]}..." if len(response.text) > 100 else f"Response text: {response.text}")
                        
                        # Try with next key
                        current_key = self._get_next_key()
                        if current_key is None:
                            logging.error("All API keys exhausted")
                            break
                        logging.info(f"Trying with next key: {current_key[:4]}...")
                        continue
                
                except requests.exceptions.RequestException as e:
                    logging.error(f"Request error: {e} for {query_date_str}")
                    # Network error, try a different key
                    current_key = self._get_next_key()
                    if current_key is None:
                        logging.error("All API keys exhausted")
                        break
                    continue
                
                retries -= 1
            
            # Add a pause every few dates to avoid rate limiting
            if query_date.day % 5 == 0 and query_date != query_date_range[-1]:
                logging.info("Pausing for 10 seconds to avoid rate limiting...")
                time.sleep(10)
        
        return data

    def run(self, start_date, end_date, savefile):
        data = self._fetch_data(start_date, end_date)
        if not data:
            logging.warning("No data was collected. Check your date range or API keys.")
            return False
            
        cwd = os.getcwd()
        savefile = os.path.join(cwd, savefile)
        with open(savefile, 'w') as f:
            json.dump(data, f)
        logging.info(f"Data saved to {savefile}")
        return True

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

    def fit(self):
        with open(self.json_path, 'r') as f:
            data = json.load(f)

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

# Example usage
if __name__ == "__main__":
    # Get data
    getter = getData(symbol='SPY')
    getter.run('2023-01-01', '2023-01-10', 'spy_options_data_test.json')
    
    # Process data
    # preprocess = preprocessData('spy_options_data_test.json')
    # df = preprocess.fit()
    # print(df.head())
