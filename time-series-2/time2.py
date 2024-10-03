import pandas as pd
from statsmodels.tsa.arima import model
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.stattools import adfuller
from scipy.stats import boxcox
from sklearn.metrics import mean_squared_error

df = pd.read_csv('AirPassengers.csv',parse_dates=['Month'],index_col=['Month'])

col, lamb = boxcox(df['#Passengers'])
df['#Passengers'] = col

plt.plot(df.diff().dropna())
plot_acf(df.dropna());
plot_pacf(df.dropna());
#p = 1
#q = 0

m = model.ARIMA(df,order=(1,0,0))

fitted = m.fit()
preds = fitted.predict(start=df.index[0],end=df.index[-1])
rmse = mean_squared_error(df,preds,squared=False)
#rmse = 0.2793075783203607