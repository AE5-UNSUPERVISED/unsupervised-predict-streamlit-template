"""

    Collaborative-based filtering for item recommendation.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within the root of this repository for guidance on how to use
    this script correctly.

    NB: You are required to extend this baseline algorithm to enable more
    efficient and accurate computation of recommendations.

    !! You must not change the name and signature (arguments) of the
    prediction function, `collab_model` !!

    You must however change its contents (i.e. add your own collaborative
    filtering algorithm), as well as altering/adding any other functions
    as part of your improvement.

    ---------------------------------------------------------------------

    Description: Provided within this file is a baseline collaborative
    filtering algorithm for rating predictions on Movie data.

"""

# Script dependencies
import pandas as pd
import numpy as np
import pickle
import copy
from surprise import Reader, Dataset
from surprise import SVD # NormalPredictor, BaselineOnly, KNNBasic, NMF
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import re

# Importing data
#movies_df = pd.read_csv('/home/explore-student/unsupervised-predict-streamlit-template/resources/data/movies.csv')
#ratings_df = pd.read_csv('/home/explore-student/unsupervised-predict-streamlit-template/resources/data/ratings.csv')
#ratings_df.drop(['timestamp'], axis=1,inplace=True)
movies_df = pd.read_csv('resources/data/movies.csv')
ratings_df = pd.read_csv('resources/data/ratings.csv')
ratings_df.drop(['timestamp'], axis=1,inplace=True)

# We make use of an SVD model trained on a subset of the MovieLens 10k dataset.
#model=pickle.load(open('/home/explore-student/unsupervised-predict-streamlit-template/resources/models/SVD.pkl', 'rb'))
model=pickle.load(open('resources/models/SVD.pkl', 'rb'))



def prediction_item(item_id):
    """Map a given favourite movie to users within the
       MovieLens dataset with the same preference.

    Parameters
    ----------
    item_id : int
        A MovieLens Movie ID.

    Returns
    -------
    list
        User IDs of users with similar high ratings for the given movie.

    """
    # Data preprosessing
    reader = Reader(rating_scale=(0.5, 5))
    load_df = Dataset.load_from_df(ratings_df,reader)
    a_train = load_df.build_full_trainset()

    predictions = []
    for ui in a_train.all_users():
        predictions.append(model.predict(iid=item_id,uid=ui, verbose = False))
    return predictions

def pred_movies(movie_list):
    """Maps the given favourite movies selected within the app to corresponding
    users within the MovieLens dataset.

    Parameters
    ----------
    movie_list : list
        Three favourite movies selected by the app user.

    Returns
    -------
    list
        User-ID's of users with similar high ratings for each movie.

    """
    # Store the id of users
    id_store=[]
    # For each movie selected by a user of the app,
    # predict a corresponding user within the dataset with the highest rating
    for i in movie_list:
        predictions = prediction_item(item_id = i)
        predictions.sort(key=lambda x: x.est, reverse=True)
        # Take the top 10 user id's from each movie with highest rankings
        for pred in predictions[:100]:
            id_store.append(pred.uid)
    # Return a list of user id's
    return id_store

# !! DO NOT CHANGE THIS FUNCTION SIGNATURE !!
# You are, however, encouraged to change its content.  
def collab_model(movie_list,top_n=10):
    """Performs Collaborative filtering based upon a list of movies supplied
       by the app user.

    Parameters
    ----------
    movie_list : list (str)
        Favorite movies chosen by the app user.
    top_n : type
        Number of top recommendations to return to the user.

    Returns
    -------
    list (str)
        Titles of the top-n movie recommendations to the user.

    """
    # Create new feature year
    movies_df['year'] = movies_df['title'].apply(lambda st: st[st.find("(")+1:st.find(")")])
    
    movies = movies_df
    # extract year from movie list
    movie_year = [x[-6:] for x in movie_list]
    # clean year
    def clean_year(movie_year):
        em = []
        for i in range(len(movie_year)):
            x = re.sub(r'[\(\)]', '', movie_year[i])
            em.append(x)
        return em
    cleanyear = clean_year(movie_year)
    # drop rows that does not have year
    index_names = movies[movies['year'].str.len() != 4].index
    movies.drop(index_names, inplace=True)
    # filter out years that are not in range
    minyear = str(int(min(cleanyear)) - 5)
    maxyear = str(int(max(cleanyear)) + 5)
    movies = movies[(movies['year'] > minyear) & (movies['year'] < maxyear)]
    
    # map the given movies selected within the app to users within the MovieLens dataset, return user ID's of users with
    # similar high ratings for each movie
    movie_ids = pred_movies(movie_list)
    df_init_users = ratings_df[ratings_df['userId']==movie_ids[0]]
    for i in movie_ids[1:]:
        df_init_users=df_init_users.append(ratings_df[ratings_df['userId']==i])
    # Getting the cosine similarity matrix
    cosine_sim = cosine_similarity(df_init_users)
    # Get the title of each movie as the index
    indices = pd.Series(movies['title'])
    # Getting the index of the movie that matches the title
    idx_1 = indices[indices == movie_list[0]].index[0]
    idx_2 = indices[indices == movie_list[1]].index[0]
    idx_3 = indices[indices == movie_list[2]].index[0]
    # Get similarity score for each index
    rank_1 = cosine_sim[idx_1]
    rank_2 = cosine_sim[idx_2]
    rank_3 = cosine_sim[idx_3]
    # Creating a Series with the similarity scores in descending order
    score_series_1 = pd.Series(rank_1).sort_values(ascending = False)
    score_series_2 = pd.Series(rank_2).sort_values(ascending = False)
    score_series_3 = pd.Series(rank_3).sort_values(ascending = False)
     # Append all series and then sort similarity scores in descending order
    listings = score_series_1.append(score_series_2).append(score_series_3).sort_values(ascending = False)
    # Initializing the empty list of recommended movies
    recommended_movies = []
    # Choose top 50 indexes of our series as a list
    top_50_indexes = list(listings.iloc[:50].index)
    # Removing chosen movies
    top_indexes = np.setdiff1d(top_50_indexes,[idx_1,idx_2,idx_3])
    # Get top 10 movies that are most similar to the users chosen 3 movies
    for i in top_indexes[:top_n]:
        recommended_movies.append(list(movies_df['title'])[i])
    return recommended_movies
