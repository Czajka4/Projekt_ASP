import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf, adfuller
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from arch import arch_model


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
