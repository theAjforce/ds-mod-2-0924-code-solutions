import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import pickle
from Source_code import *

if __name__ == "__main__":
    df = file_reader("/Users/anthonyvillegas/Documents/GitHub/ds-mod-2-0924-code-solutions/recommendation-systems-project/goodreads_reviews_poetry.json")
    filtered_df = filterer(df)
    with open("filtered_df.pickle",'wb') as f:
        pickle.dump(filtered_df,f)
    utility_matrix = utility_maker(filtered_df)
    with open("utility_matrix.pickle",'wb') as f:
        pickle.dump(utility_matrix,f)
    corr_matrix = corr_maker(utility_matrix)
    with open("corr_matrix.pickle",'wb') as f:
        pickle.dump(corr_matrix,f)
    nn_model = nn_maker(corr_matrix)
    with open("nn_model.pickle",'wb') as f:
        pickle.dump(nn_model,f)