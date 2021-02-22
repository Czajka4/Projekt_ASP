from datetime import datetime, timedelta
import time

import requests
import  lxml
from lxml import html

import numpy as np
import pandas as pd

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
              "accept": "text/html",
              "accept-encoding": "gzip, deflate, br",
              "accept-language": "en-US,en;q=0.9",
              "cache-control": "no-cache",
              "dnt": "1",
              "pragma": "no-cache",
              "sec-fetch-mode": "navigate",
              "sec-fetch-site": "same-origin",
              "sec-fetch-user": "?1",
              "upgrade-insecure-requests": "1",
              "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64)"}
     
     return hdrs

def scrape_yahoo():
    symbol = 'MSFT'

    dt_start = datetime.today() - timedelta(days=365)
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

    element_html = html.fromstring(page.content)

    table = element_html.xpath('//table')

    table_tree = lxml.etree.tostring(table[0], method='xml')

    panda = pd.read_html(table_tree)

    df = pd.DataFrame(panda[0])

    df = df[~df['Close*'].str.contains('Dividend')]
    df = df[~df['Close*'].str.contains('Close')]     

    closing = np.array(df['Close*'].astype(float).values)

    return closing