from sklearn.decomposition import PCA
import pandas as pd

df = pd.read_csv('mnist.csv')
df.drop('label',axis=1,inplace=True)
pca = PCA(n_components=84)
pca.components_