# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
np.random.seed(42)

df = pd.read_csv('btc_2h.csv')
# X = df.iloc[:, 0:-1].values
ts=df.iloc[:, -1].values
# X.shape
# y[0]=y[1]
x=0
y=[None]*(len(df)-52)
for i in range(len(df)-52):
  y[x]=ts[52+i]-ts[i]
  x=x+1
x

y[0]=y[1]

plt.plot(ts)
plt.grid()
# plt.tight_layout()
plt.show()

plt.plot(y)
plt.grid()
plt.tight_layout()
plt.show()

from statsmodels.tsa.stattools import adfuller

def test_stationarity(timeseries):

    #Determing rolling statistics
    rolmean = timeseries.rolling(window=52).mean()
    rolstd = timeseries.rolling(window=52).std()
    #Plot rolling statistics:
    plt.plot(timeseries, color='blue',label='Original')
    plt.plot(rolmean, color='red', label='Rolling Mean')
    plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.grid()
    plt.show()
    #Perform Dickey-Fuller test:
    print ('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)

    critical_value = dftest[4]['5%']
    test_statistic = dftest[0]
    alpha = 1e-3
    pvalue = dftest[1]
    if pvalue < alpha and test_statistic < critical_value:  # null hypothesis: x is non stationary
        print("X is stationary")
        return True
    else:
        print("X is not stationary")
        return False

ts_diff = pd.Series(y)
d = 0
while test_stationarity(ts_diff) is False:
    ts_diff = ts_diff.diff().dropna()
    d = d + 1

d

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
plot_acf(y, lags =25)
# plt.savefig('plots/acf.png')
plt.show()
plot_pacf(y, lags =12)
# plt.savefig('plots/pacf.png')
plt.show()

from statsmodels.tsa.statespace.sarimax import SARIMAX

p = 0
q = 0
l=0
r=[None]*(len(y)-150)
for i in range(0,len(y)-150):
  r[l]=y[i]
  l=l+1
model = SARIMAX(r, order=(p,d,q))
model_fit = model.fit(disp=1,solver='powell')

fcast = model_fit.get_prediction(start=len(y)-149, end=len(y),dynamic=True )
ts_p = fcast.predicted_mean
ts_ci = fcast.conf_int()

z=[None]*150
j=0
for i in range(len(y)-150,len(y)):
  z[j]=y[i]
  j=j+1
plt.show()
plt.plot(ts_p,label='prediction')
plt.plot(z,color='red',label='actual')
# plt.fill_between(ts_ci.index[1:],
#                 ts_ci.iloc[1:, 0],
#                 ts_ci.iloc[1:, 1], color='k', alpha=.2)

plt.ylabel('Total Number of Tourists Arrivals')
plt.legend()
plt.tight_layout()
plt.grid()
# plt.savefig('plots/IT_trend_prediction.png')
plt.show()

from statsmodels.tsa.statespace.sarimax import SARIMAX
X = y
size = int(len(X) * 0.93)


train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = []
p=23
d=1
q=9

for t in range(len(test)):
	model = SARIMAX(history, order=(p,d,q))
	model_fit = model.fit(disp=1)
	output = model_fit.forecast()
	yhat = output[0]
	predictions.append(yhat)
	obs = test[t]
	history.append(obs)
	print('predicted=%f, expected=%f' % (yhat, obs))

plt.plot(predictions,label='prediction')
plt.plot(test,color='red',label='actual')
# plt.fill_between(ts_ci.index[1:],
#                 ts_ci.iloc[1:, 0],
#                 ts_ci.iloc[1:, 1], color='k', alpha=.2)

plt.ylabel('Total Number of Tourists Arrivals')
plt.legend()
plt.tight_layout()
plt.grid()
# plt.savefig('plots/IT_trend_prediction.png')
plt.show()

from sklearn.metrics import r2_score
r2_score(test, predictions)

import math
MSE = np.absolute(np.subtract(test,predictions)).mean()
print("Mean Absolute Error:\n")
print(MSE)

