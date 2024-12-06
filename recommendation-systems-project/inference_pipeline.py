import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import pickle
from Source_code import *

if __name__ == "__main__":
    pipeline_user_input = get_user()
    filtered_df = pickle.load(open("filtered_df.pickle","rb"))
    utility_matrix = pickle.load(open("utility_matrix.pickle","rb"))
    corr_matrix = pickle.load(open("corr_matrix.pickle","rb"))
    nn_model = pickle.load(open("nn_model.pickle","rb"))
    print(recommend_books(pipeline_user_input,utility_matrix,filtered_df,corr_matrix,nn_model))