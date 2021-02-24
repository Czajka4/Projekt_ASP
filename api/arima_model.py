import numpy as np
import pandas as pd 
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf, adfuller
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from datetime import timedelta


def calculate_arima(df):    
    df_fill = df.asfreq('W', method='ffill')
    df_returns = df_fill.diff()
    df_returns.iloc[0] = 0

    #Build the model and fit it to the data
    model = ARIMA(df_returns, order=[1,0,1])
    fitted_model = model.fit()

    #debug
    #print(fitted_model.summary())
    #print(acorr_ljungbox(fitted_model.resid,lags=[1,2,3,4,5])[1])

    start = df.iloc[[-1]].index
    end = start + timedelta(weeks=10)

    start = pd.to_datetime(str(start.values[0]))
    end =  pd.to_datetime(str(end.values[0]))

    start_date = start.strftime('%Y.%m.%d')
    end_date = end.strftime('%Y.%m.%d')

    preds = fitted_model.predict(start=start_date,end=end_date)
    dfp = pd.DataFrame(preds)
    dfp = dfp.rename(columns={"predicted_mean": "price"})

    last_v = df_fill.price.iloc[-1]
    for n, x in enumerate(dfp.price.iloc[:]):
        if n < 1:
            dfp.price[n] = last_v + x
        else:
            dfp.price[n] = dfp.price.iloc[n-1] + dfp.price.iloc[n]

    df_fill.price = df_fill.price.fillna(0)
    dfp.price = dfp.price.fillna(0)

    for n, x in enumerate(df_fill.price.iloc[:]):
        if n < 1:
            if x < 0.0001:
                df_fill.price[n] = df_fill.price[n+1]
        else:
             if x < 0.0001:
                df_fill.price[n] = df_fill.price[n-1]

    return df_fill, dfp


 