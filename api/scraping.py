from datetime import datetime, timedelta
import time
import re
import json

import requests
import  lxml
from lxml import html

import numpy as np
import pandas as pd

from arima import calculate_arima


def format_data_frame(df):
    df = df.rename(columns={"Date": "date", "close": "price"})
    df = df.drop(df.columns.difference(['date','price']), 1)

    df.price = df.price.astype(float)

    df.date = pd.to_datetime(df.date, unit='s')
    df.date = df.date.dt.strftime('%Y-%m-%d')

    df = df.set_index('date')
    df.index = pd.DatetimeIndex(df.index).to_period('D')
    df.index = df.index.to_timestamp()
    df = df[~df.index.duplicated()]
    df = df.reindex(pd.date_range(start=df.index.min(), end=df.index.max(), freq='W-FRI'))
    
    return df


def format_date(date_datetime):
     date_timetuple = date_datetime.timetuple()
     date_mktime = time.mktime(date_timetuple)
     date_int = int(date_mktime)
     date_str = str(date_int)
     return date_str


def subdomain(symbol, start, end, filter='history'):
     subdoma="/quote/{0}/history?period1={1}&period2={2}&interval=1d&filter={3}&frequency=1d"
     subdomain = subdoma.format(symbol, start, end, filter)
     return subdomain


def header_function(subdomain):
     hdrs =  {"authority": "finance.yahoo.com",
              "method": "GET",
              "path": subdomain,
              "scheme": "https",              
              "accept-encoding": "gzip, deflate, br",
              "accept-language": "en-US,en;q=0.9",
              "cache-control": "no-cache",
              "dnt": "1",
              "pragma": "no-cache",
              "sec-fetch-mode": "navigate",
              "sec-fetch-site": "same-origin",
              "sec-fetch-user": "?1",
              "upgrade-insecure-requests": "1"
              }
     
     return hdrs


def scrape_yahoo():
    symbol = 'MSFT'

    #for n in range()
    dt_start = datetime.today() - timedelta(days= 730)   
    dt_end = datetime.today() 
    
    start = format_date(dt_start)
    end = format_date(dt_end)
    
    sub = subdomain(symbol, start, end)
    header = header_function(sub)
    print(header)
    base_url = 'https://finance.yahoo.com'
    url = base_url + sub
    page = requests.get(url)

    print(f"\n {url} \n")
    print(f"\t >>>>   {page.status_code} \n ")

    p = re.compile('HistoricalPriceStore":{"prices":(.*?\])')
    data = json.loads(p.findall(page.text)[0])

    df = pd.DataFrame(data)
    df = format_data_frame(df)

    df, dfp = calculate_arima(df)

    return df, dfp


if __name__ == "__main__":
     scrape_yahoo()
