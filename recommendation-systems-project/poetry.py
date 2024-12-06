import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from scipy.sparse.linalg import svds
import pickle

df = pd.read_json('/Users/anthonyvillegas/Documents/GitHub/ds-mod-2-0924-code-solutions/recommendation-systems-project/goodreads_reviews_poetry.json',lines=True)
#df = pd.read_parquet("/Users/anthonyvillegas/Documents/GitHub/ds-mod-2-0924-code-solutions/recommendation-systems-project/poetry.parquet",columns=['user_id','book_id','rating'])


#filtering user's based on books rated
book_counts = df['book_id'].value_counts()
filtered_books = book_counts[book_counts >= 100].index
filtered_df = df[df['book_id'].isin(filtered_books)]

#utility matrix
um_poet = filtered_df.pivot_table(index='book_id',columns='user_id',values='rating').fillna(0)
#utility matrix for the user_ratings_and_books dictionary
um_poet2 = filtered_df.pivot_table(index='user_id',columns='book_id',values='rating').fillna(0)
#correlation matrix, have to fillna(0) because when creating the correlation matrix um_corr creates NaNs
um_corr = um_poet.T.corr().fillna(0)
#cosine similarity matrix
cs_poet = cosine_similarity(um_poet)
cs_poet = pd.DataFrame(cs_poet,columns=um_poet.T.columns,index=um_poet.T.columns)

#input for functions
user = input("Enter a user ID: ")
#item to item recommendation with user input
def recommend_books(user):
    user_ratings = um_poet[user]
    num_ratings_above_3 = (user_ratings >= 3).sum()
    rated_items = filtered_df[filtered_df['user_id'] == user]['book_id'].tolist()

    if num_ratings_above_3 >= 5: #Only looking at user's that have 5 or more positive ratings
        nn = NearestNeighbors(n_neighbors=4)
        nn.fit(um_corr)
        user_ratings_and_books = {}
        for col in um_poet:
            user_ratings_and_books[col] = um_poet[col].tolist() #This creates a dictionary that we will use as our linear combination to multiply down axis 0 of the correlation matrix to find the best recommendations
        new_matrix = pd.DataFrame((user_ratings_and_books[user] * um_corr).sum(axis=0))
        distance, indices = nn.kneighbors(new_matrix.T)
        poetry_books = um_poet.T.columns[indices[0][1:4]].tolist()
        filtered_recommendations = [item for item in poetry_books if item not in rated_items]
        return poetry_books
    else:  #If they have fewer than 5 positive ratings they are given a generic recommendation list of items with the highest average rating.
        average_ratings = filtered_df.groupby('book_id')['rating'].mean()
        sorted_books = average_ratings.sort_values(ascending=False)
        recommendations = sorted_books.head(3).index.tolist()
        filtered_recommendations = [item for item in recommendations if item not in rated_items]
        return filtered_recommendations


#function to get U,sigma,Vt,and um_mean to create the svd reduced matrix
def svd_values():
    R = um_poet.values
    um_mean = np.mean(R,axis=1)
    um_mean = um_mean.reshape(-1,1)
    R_demeaned = R - um_mean
    U, sigma, Vt = svds(R_demeaned,k=4)
    sigma = np.diag(sigma)
    return U,sigma,Vt,um_mean
U,sigma,Vt,um_mean = svd_values()
#correlation matrix svd reduced
def new_corr_matrix():
    um_reconstructed = U @ sigma @ Vt
    um_reconstructed = um_reconstructed + um_mean
    new_corr = np.corrcoef(um_reconstructed)
    new_corr = np.nan_to_num(new_corr)
    return new_corr,um_reconstructed
new_corr,um_reconstructed = new_corr_matrix()
#creating a new nn model for svd reduced
def create_nn_svd():
    nn_svd = NearestNeighbors(n_neighbors=4)
    nn_svd.fit(new_corr)
    return nn_svd
nn_svd = create_nn_svd()
#getting top 4 recommended books using svd reduced matrix
def reco_from_svd(user):
    ratings_vector = np.array(um_poet[user])
    new_matrix = pd.DataFrame((ratings_vector * new_corr).sum(axis=0))
    distance, indices = nn_svd.kneighbors(new_matrix.T)
    poetry_books = um_poet.T.columns[indices[0][1:4]].tolist()
    return poetry_books

#function to calculate rmse of both utility matrices
def calc_rmse_matrix(um_poet,um_reconstructed):
    squared_diffs = (um_poet - um_reconstructed)**2
    mean_squared_diffs = np.mean(squared_diffs) 
    rmse = np.sqrt(mean_squared_diffs)
    return rmse

#printing the output for functions
print(recommend_books(user))
print(reco_from_svd(user))
print(calc_rmse_matrix(um_poet,um_reconstructed))