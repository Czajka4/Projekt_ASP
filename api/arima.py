import numpy as np
import pandas as pd 
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf, adfuller
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from arch import arch_model



def calculate_arima(df):    
    #dpr_stocks_fill = cdpr_stocks.asfreq('B',method='ffill')
   
    #Turning price time series to a returns time series    

    df_fill = df.asfreq('W', method='ffill')
    df_returns = df_fill.diff()
    df_returns.iloc[0] = 0



    #Build the model and fit it to the data
    model = ARIMA(df_returns, order=[1,0,1])
    fitted_model = model.fit()
    print(fitted_model.summary())

    print(acorr_ljungbox(fitted_model.resid,lags=[1,2,3,4,5])[1])
    preds = fitted_model.predict(start='2020-01-05',end='2020-04-05')
    
    print(preds)