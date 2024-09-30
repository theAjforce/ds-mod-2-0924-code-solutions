import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
#1. the optimal vslue of k appears to be between 3 and 4
data_df = pd.read_csv('/content/imports-85.data')
data_df.replace('?', 0, inplace=True)
x = data_df[['13495','111']]
scaler = MinMaxScaler()
x_scaled = pd.DataFrame(scaler.fit_transform(x),columns=x.columns)
inertia = []
s_score = []
c_score = []

for k in range(2,11):
  km = KMeans(n_clusters=k,n_init=20)
  km.fit(x_scaled)
  inertia.append(km.inertia_)
  s_score.append(silhouette_score(x_scaled,km.labels_))
  c_score.append(calinski_harabasz_score(x_scaled,km.labels_))

plt.plot(range(2,11),inertia)
plt.plot(range(2,11),s_score)
plt.plot(range(2,11),c_score)
km = KMeans(n_clusters=3,n_init=20)
km.fit(x_scaled)
plt.scatter(x_scaled['13495'],x_scaled['111'],c=km.labels_)
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],marker='x',c='r')
#2.
dbs = DBSCAN(eps=0.125,min_samples=5)
dbs.fit(x_scaled)
plt.scatter(x_scaled['13495'],x_scaled['111'],c=dbs.labels_)
#3.
mlb_df = pd.read_csv('/content/mlb_batting_cleaned.csv')

def nearest_players(player):
  df = mlb_df[mlb_df['Name'] == player]
  df = df.drop(['Name','Tm','Lg'],axis=1)
  x = mlb_df.drop(['Name','Tm','Lg'],axis=1)
  scaler = MinMaxScaler()
  x_scaled = pd.DataFrame(scaler.fit_transform(x),columns=x.columns)
  nbrs = NearestNeighbors(n_neighbors=3)
  nbrs.fit(x_scaled)
  _, indices = nbrs.kneighbors(df)
  p1 = mlb_df.iloc[indices[0][1]]['Name']
  p2 = mlb_df.iloc[indices[0][2]]['Name']
  print(f'The two nearest players to {player} are {p1} and {p2}')