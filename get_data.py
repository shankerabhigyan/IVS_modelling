import pandas as pd
import requests

api_key = "0C4JF7CVP6E14A6N"
# query_date = "2017-11-15"
symbol = "SPY"
query_date_range = pd.date_range(start='2024-01-15', end='2024-01-31')

data = []
for query_date in query_date_range:
    query_date = query_date.strftime('%Y-%m-%d')
    url = f'https://www.alphavantage.co/query?function=HISTORICAL_OPTIONS&symbol={symbol}&date={query_date}&apikey={api_key}'
    response = requests.get(url)
    d = response.json()
    print(d)
    if d['message']=='success':
        data.append(d['data'])

# save data to file
import json
with open('data_new_15_31_01_24.json', 'w') as f:
    json.dump(data, f)