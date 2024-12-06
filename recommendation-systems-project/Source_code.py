import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from scipy.sparse.linalg import svds
import pickle

def get_user():
    global pipeline_user_input
    pipeline_user_input = input("Enter a user ID: ")
    return pipeline_user_input

def file_reader(filename):
    df = pd.read_json(filename,lines=True)
    return df

def filterer(df):
    book_counts = df['book_id'].value_counts()
    filtered_books = book_counts[book_counts >= 100].index
    filtered_df = df[df['book_id'].isin(filtered_books)]
    return filtered_df

def utility_maker(filtered_df):
    utility_matrix = filtered_df.pivot_table(index='book_id',columns='user_id',values='rating').fillna(0)
    return utility_matrix

def corr_maker(utility_matrix):
    corr_matrix = utility_matrix.T.corr().fillna(0)
    return corr_matrix

def nn_maker(corr_matrix):
    nn_model = NearestNeighbors(n_neighbors=4)
    nn_model.fit(corr_matrix)
    return nn_model

def recommend_books(user,utility_matrix,filtered_df,corr_matrix,nn_model):
    user_ratings = utility_matrix[user]
    num_ratings_above_3 = (user_ratings >= 3).sum()
    rated_items = filtered_df[filtered_df['user_id'] == user]['book_id'].tolist()

    if num_ratings_above_3 >= 5: #Only looking at user's that have 5 or more positive ratings
        user_ratings_and_books = {}
        for col in utility_matrix:
            user_ratings_and_books[col] = utility_matrix[col].tolist() #This creates a dictionary that we will use as our linear combination to multiply down axis 0 of the correlation matrix to find the best recommendations
        new_matrix = pd.DataFrame((user_ratings_and_books[user] * corr_matrix).sum(axis=0))
        distance, indices = nn_model.kneighbors(new_matrix.T)
        poetry_books = utility_matrix.T.columns[indices[0][1:4]].tolist()
        filtered_recommendations = [item for item in poetry_books if item not in rated_items]
        return poetry_books
    
    else:  #If they have fewer than 5 positive ratings they are given a generic recommendation list of items with the highest average rating.
        average_ratings = filtered_df.groupby('book_id')['rating'].mean()
        sorted_books = average_ratings.sort_values(ascending=False)
        recommendations = sorted_books.head(3).index.tolist()
        filtered_recommendations = [item for item in recommendations if item not in rated_items]
        return filtered_recommendations