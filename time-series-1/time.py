import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.stattools import adfuller

df = pd.read_csv('AirPassengers.csv')

df['Month'] = pd.to_datetime(df['Month'])

df.set_index('Month',inplace=True)

df['rolling mean'] = df['#Passengers'].rolling(window=12).mean()
df['rolling std'] = df['#Passengers'].rolling(window=12).std()
plt.plot(df['#Passengers'],color='blue')
plt.plot(df['rolling mean'],color='red')
plt.plot(df['rolling std'],color='green')

df['new passengers'] = df['#Passengers'].shift()
df['diff1'] = df['#Passengers']-df['new passengers']
df['new rm'] = df['diff1'].rolling(window=12).mean()
df['new std'] = df['diff1'].rolling(window=12).std()
plt.plot(df['diff1'],color='blue')
plt.plot(df['new rm'],color='red')
plt.plot(df['new std'],color='green')

df['pass log'] = np.log(df['#Passengers'])
plt.plot(df['pass log'])

adf = adfuller(df['#Passengers'])
#p-value:0.9918802434376411
#not stationary