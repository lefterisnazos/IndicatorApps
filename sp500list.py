import pandas as pd

def get_sp500_tickers():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    tables = pd.read_html(url)
    df = tables[0]
    return df['Symbol'].tolist()

sp500_list = get_sp500_tickers()
print(sp500_list)