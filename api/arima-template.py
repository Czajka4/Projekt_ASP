import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf, adfuller
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from arch import arch_model
from datetime import timedelta

if __name__ == "__main__":
    
    cdpr_stocks = pd.read_csv('cdpr.csv',
                              index_col='Date',
                              squeeze=True,
                              usecols=['Date','Close'], 
                              parse_dates=True)
    cdpr_stocks = cdpr_stocks.rename_axis('Date')
    cdpr_stocks = cdpr_stocks.rename('Price')

    plt.plot(cdpr_stocks['2019-05'],label='CDPR stocks', marker='.',linestyle='-')
    plt.ylabel('Cena')
    plt.title('Wykres cen akcji spółki CD Projekt S.A.')
    plt.xticks(rotation=45)
    plt.axhline(np.mean(cdpr_stocks['2019-05']),color='r')
    plt.axhline(np.quantile(cdpr_stocks['2019-05'],0.25),color='grey',linestyle='dashed')
    plt.axhline(np.quantile(cdpr_stocks['2019-05'],0.75),color='grey',linestyle='dashed')
    plt.show()
    
    print(cdpr_stocks.head())
    
    
    cdpr_stocks_fill = cdpr_stocks.asfreq('B',method='ffill')
    
    #Decomposition of a time series
    cdpr_stocks_decomposed = seasonal_decompose(cdpr_stocks_fill['2019'],model='additive')
    cdpr_stocks_decomposed.plot()
    

    #Turning price time series to a returns time series
    
    cdpr_stocks_returns = cdpr_stocks_fill.diff()
    cdpr_stocks_returns.iloc[0] = 0
    
#    Optional comparison of time series
#    comparison_full_fig, axis = plt.subplots(3)
#    axis[0].hist(cdpr_stocks_fill, align='mid', edgecolor='k')
#    axis[1].hist(np.log(cdpr_stocks_fill), align='mid', edgecolor='k')
#    axis[2].hist(cdpr_stocks_returns, align='mid', edgecolor='k')
#    axis[0].set_title("Price")
#    axis[1].set_title("log(Price)")
#    axis[2].set_title("Returns")
#    plt.show(comparison_full_fig)

    #Check PACF and ACF plots
    plot_acf(cdpr_stocks_returns,lags=20)
    plot_pacf(cdpr_stocks_returns,lags=20)
    
    #Build the model and fit it to the data
    model = ARIMA(cdpr_stocks_returns,order=[1,0,1])
    fitted_model = model.fit()
    print(fitted_model.summary())
    
    #Check residuals
    print(acorr_ljungbox(fitted_model.resid,lags=[1,2,3,4,5])[1])
    plot_acf(fitted_model.resid,lags=30)
    
    #Get prediction from the fitted ARIMA model
    preds = fitted_model.predict(start='2019-01-02',end='2020-12-31')
#   preds = fitted_model.forecast(steps=10)
    print(preds)
    plt.plot(preds,color='orange')
    plt.plot(cdpr_stocks_returns, color='grey')
    
        #Build and fit an ARCH model and get a prediction
    am = arch_model(cdpr_stocks_returns)
    res = am.fit()
    forecasts = res.forecast(horizon=5)
    print(forecasts.variance.iloc[-3:])
   '''
    217]: %history
df
url =  https://finance.yahoo.com/quote/MSFT/history?period1=1551034636&period2=1614106636&interval=1d&filter=history&frequency=1d
url = ' https://finance.yahoo.com/quote/MSFT/history?period1=1551034636&period2=1614106636&interval=1d&filter=history&frequency=1d'
page = requests.get(url)
    print(f"\n {url} \n")
    print(f"\t >>>>   {page.status_code} \n ")
    #element_html = html.fromstring(page.content)
    #table = element_html.xpath('//table')
    #table_tree = lxml.etree.tostring(table[0], method='xml')
    p = re.compile('HistoricalPriceStore":{"prices":(.*?\])')
    data = json.loads(p.findall(page.text)[0])
    df = pd.DataFrame(data)
    df = df.rename(columns={"Date": "date", "close": "price"})
    df = df.drop(df.columns.difference(['date','price']), 1)
    df.price = df.price.astype(float)
    df.date = pd.to_datetime(df.date, unit='s')
    df.date = df.date.dt.strftime('%Y-%m-%d')
    #df = df.asfreq('W-FRI', method='pad')
 page = requests.get(url)
from datetime import datetime, timedelta
import time
import re
import json
import requests
import  lxml
from lxml import html
import numpy as np
import pandas as pd
page = requests.get(url)
 p = re.compile('HistoricalPriceStore":{"prices":(.*?\])')
    data = json.loads(p.findall(page.text)[0])
    df = pd.DataFrame(data)
    df = df.rename(columns={"Date": "date", "close": "price"})
    df = df.drop(df.columns.difference(['date','price']), 1)
    df.price = df.price.astype(float)
    df.date = pd.to_datetime(df.date, unit='s')
    df.date = df.date.dt.strftime('%Y-%m-%d')
    #df = df.asfreq('W-FRI', method='pad')
    print(df)
dfc = df.copy()
df.reindex(pd.date_range(start=sp.index.min(), end=sp.index.max(), freq='W-FRI' ))
df.reindex(pd.date_range(start=df.index.min(), end=df.index.max(), freq='W-FRI' ))
df
df = df.set_index('date')
 df.index = pd.DatetimeIndex(df.index).to_period('D')
df
df.reindex(pd.date_range(start=df.index.min(), end=df.index.max(), freq='W-FRI' ))
df.index
df.index.to_timestamp
df.index[0]
df..asfreq('W-FRI', method='pad')
df.asfreq('W-FRI', method='pad')
df.asfreq('W-FRI', method='ffill')
df.asfreq('W-FRI')
df.reindex(pd.date_range(start=df.index.min(), end=df.index.max(), freq='W-FRI' ))
df.index.to_timestamp()
df.index = df.index.to_timestamp()
df
df.reindex(pd.date_range(start=df.index.min(), end=df.index.max(), freq='W-FRI' ))
df
df[df.index.duplicated()]
df = df[~df.index.duplicated()]
df.reindex(pd.date_range(start=df.index.min(), end=df.index.max(), freq='W-FRI' ))
%history
 df_returns = df.diff()
 df_returns.iloc[0] = 0
df_returns
model = ARIMA(df_returns, order=[1,0,1])
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf, adfuller
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from arch import arch_model
model = ARIMA(df_returns, order=[1,0,1])
df.asfreq('B', method='ffill')
df
df.index = pd.DatetimeIndex(df.index).to_period('D')
df
df_returns = df.diff()
 df_returns.iloc[0] = 0
 model = ARIMA(df_returns, order=[1,0,1])
df.index = pd.DatetimeIndex(df.index).to_period('W')
 df
dfc
df = dfc.copy()
df.price = df.price.astype(float)
    df.date = pd.to_datetime(df.date, unit='s')
    df.date = df.date.dt.strftime('%Y-%m-%d')
    #df = df.asfreq('W-FRI', method='pad')
    df = df.set_index('date')
    df.index = pd.DatetimeIndex(df.index).to_period('D')
    df.index = df.index.to_timestamp()
    df = df[~df.index.duplicated()]
    df = df.reindex(pd.date_range(start=df.index.min(), end=df.index.max(), freq='W-FRI'))
df
df.index
df.reindex(pd.date_range(start=df.index.min(), end=df.index.max(), freq='W-FRI'))
df.reindex(pd.date_range(start=df.index.min(), end=df.index.max(), freq='B'))
df
df.asfreq('B')
df.asfreq('B', method='ffill')
df.asfreq('W', method='ffill')
df
df_fill = df.asfreq('W', method='ffill')
 df_returns = df_fill.diff()
    df_returns.iloc[0] = 0
df_returns
 model = ARIMA(df_returns, order=[1,0,1])
f\itted_model = model.fit()
fitted_model = model.fit()
 print(fitted_model.summary())
    print(acorr_ljungbox(fitted_model.resid,lags=[1,2,3,4,5])[1])
 preds = fitted_model.predict(start='2020-01-02',end='2020-04-01')
 preds = fitted_model.predict(start='2020-01-03',end='2020-04-01')
 preds = fitted_model.predict(start='2020-01-04',end='2020-04-01')
 preds = fitted_model.predict(start='2020-01-05',end='2020-04-01')
 preds = fitted_model.predict(start='2020-01-05',end='2020-04-02')
 preds = fitted_model.predict(start='2020-01-05',end='2020-04-03')
 preds = fitted_model.predict(start='2020-01-05',end='2020-04-4')
 preds = fitted_model.predict(start='2020-01-05',end='2020-04-05')
predds
preds
am = arch_model(df_returns)
res = am.fit()
forecasts = res.forecast(horizon=5)
print(forecasts.variance.iloc[-3:])
%history
df
df.iloc[0. -1]
df.iloc[[-1]]
df.iloc[[-1]].index
df.iloc[[-1]].index.to_datetime
df.iloc[[-1]].index.to_datetime()
df.iloc[[-1]].index
df.iloc[[-1]].index.value
df.iloc[[-1]].index
XZ = df.iloc[[-1]].index
x = XZ
x
x[0]
x[1]
x[0]
x[0][0]
str(x)
x.value
pd.to_datetime(x)
x
 from datetime import timedelta
x + timedelta(weeks=1)
x + timedelta(weeks=2)
x + timedelta(weeks=3)
df
df_returns
result = {}
prev
pred
preds
            x
df
weeks = []
weeks = [x + timedelta(weeks=1+n) for n range(10)]
x
 [x + timedelta(weeks=1+n) for n range(10)]
 [x + timedelta(weeks=1+n) for n in range(10)]
weeks = [x + timedelta(weeks=1+n) for n in range(10)]
preds = fitted_model(weeks)
x
str(x[0])
 fitted_model(start=str(x[0]), n.ahead=10)
 fitted_model(start=str(x[0]), end=str(x[0] + timedelta(weels=10)))
 fitted_model(start=str(x[0]), end=str(x[0] + timedelta(weeks=10)))
preds = fitted_model.predict(start='2020-01-05',end='2020-04-05')
preds = fitted_model.predict(start='2020-01-05',end='2020-04-05')
x
x[0].strftime("%y-%m-%d")
preds = fitted_model.predict(start= x[0].strftime("%y-%m-%d"),end=( x[0] + timedelta(weeks=10).strftime("%y-%m-%d")))
x[0] + timedelta(weeks=10)
(x[0] + timedelta(weeks=10))[0]
(x[0] + timedelta(weeks=10)).to_datetime64
(x[0] + timedelta(weeks=10)).to_datetime64()
(x[0] + timedelta(weeks=10)).to_datetime64().strftime()
str((x[0] + timedelta(weeks=10)).to_datetime64())
(x[0] + timedelta(weeks=10))[0]
(x[0] + timedelta(weeks=10))
(x[0] + timedelta(weeks=10)).to_tuples()
(x[0] + timedelta(weeks=10)).to_series
(x[0] + timedelta(weeks=10)).to_array
str((x[0] + timedelta(weeks=10)).to_datetime64())
((x[0] + timedelta(weeks=10)).to_datetime64())
x
x.all()
x
x.values
x.values[0]
t= pd.to_datetime(str(date))
t = pd.to_datetime(str(x.values[0]))
t
timestring = t.strftime('%Y.%m.%d')
timestring
t
t + timedelta(weeks=10)
start_date = pd.to_datetime(str(df.iloc[[-1]].index))
start_date = pd.to_datetime(str(df.iloc[[-1]].index.values[0]))
start_date
start_date = pd.to_datetime(str(df.iloc[[-1]].index.values[0])).stftime('%Y.%m.%d')
start_date/strftime('%Y.%m.%d')
start_date.strftime('%Y.%m.%d')
start_date  = start_date.strftime('%Y.%m.%d')
end_date = pd.to_datetime(str(df.iloc[[-1]].index.values[0] + timedelta(weeks=10)))
end_date = pd.to_datetime(str(df.iloc[[-1]].index.values[0] + timedelta(weeks=10)))
z = x + timedelta(weeks=10)
z
start_date = pd.to_datetime(str(z))
%history
'''
