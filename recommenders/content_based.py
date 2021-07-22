"""

    Content-based filtering for item recommendation.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within the root of this repository for guidance on how to use
    this script correctly.

    NB: You are required to extend this baseline algorithm to enable more
    efficient and accurate computation of recommendations.

    !! You must not change the name and signature (arguments) of the
    prediction function, `content_model` !!

    You must however change its contents (i.e. add your own content-based
    filtering algorithm), as well as altering/adding any other functions
    as part of your improvement.

    ---------------------------------------------------------------------

    Description: Provided within this file is a baseline content-based
    filtering algorithm for rating predictions on Movie data.

"""

# Script dependencies
import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import re

# Importing data
#movies = pd.read_csv('/home/explore-student/unsupervised-predict-streamlit-template/resources/data/movies.csv')
#ratings = pd.read_csv('/home/explore-student/unsupervised-predict-streamlit-template/resources/data/ratings.csv')
#imdb = pd.read_csv('/home/explore-student/unsupervised-predict-streamlit-template/resources/data/imdb_data.csv')
movies = pd.read_csv('resources/data/movies.csv')
ratings = pd.read_csv('resources/data/ratings.csv')
#imdb = pd.read_csv('resources/data/imdb_data.csv')
movies.dropna(inplace=True)

# create new feature year
#movies['year'] = movies['title'].str[-6:]
#movies['year'] = movies['year'].apply(lambda st: st[st.find("(")+1:st.find(")")])

def preprocess_genre(genre):
    genre = re.sub(r'[\-]', '_', genre)
    genre = re.sub(r'[\(\)]', '', genre)
    genre = re.sub(r'no genres listed', 'no_genres_listed', genre)
    genre = ' '.join([word for word in genre.split('|')])
    return genre

#def preprocess_titlecast(title_cast):
   # title_cast = re.sub(r'\ ', '_', str(title_cast))
    #title_cast = ' '.join([word for word in title_cast.split('|')])
    #return title_cast

#def preprocess_director(director):
 #   director = re.sub(r'\ ', '_', str(director))
  #  return director

#def preprocess_keywords(keywords):
 #   keywords = ' '.join([word for word in str(keywords).split('|')])
  #  return keywords

def data_preprocessing(subset_size):
    """Prepare data for use within Content filtering algorithm.

    Parameters
    ----------
    subset_size : int
        Number of movies to use within the algorithm.

    Returns
    -------
    Pandas Dataframe
        Subset of movies selected for content-based filtering.

    """

    # Merge movies and imdb dataframes
    #movies_imdb = pd.merge(left=movies, right=imdb, how='left', on='movieId')

    # Apply preprocessing functions to genres, title cast, director and plot keywords
    movies['genres'] = movies['genres'].apply(preprocess_genre)
    #movies_imdb['title_cast'] = movies_imdb['title_cast'].apply(preprocess_titlecast)
    #movies_imdb['director'] = movies_imdb['director'].apply(preprocess_director)
    #movies_imdb['plot_keywords'] = movies_imdb['plot_keywords'].apply(preprocess_keywords)
    # Subset of the data
    movies_subset = movies[:subset_size]
    return movies_subset

# !! DO NOT CHANGE THIS FUNCTION SIGNATURE !!
# You are, however, encouraged to change its content.  
def content_model(movie_list,top_n=10):
    """Performs Content filtering based upon a list of movies supplied
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

    # extract year from movie list
    #movie_year = [x[-6:] for x in movie_list]
    # clean year
    #def clean_year(movie_year):
     #   em = []
      #  for i in range(len(movie_year)):
       #     x = re.sub(r'[\(\)]', '', movie_year[i])
        #    em.append(x)
        #return em
    #cleanyear = clean_year(movie_year)
    
    # preprocess data and select subset
    data = data_preprocessing(27000)
    # drop rows that does not have year
    #index_names = data[data['year'].str.len() != 4].index
    #data.drop(index_names, inplace=True)
    # filter out years that are not in range
    #minyear = str(int(min(cleanyear)) - 5)
    #maxyear = str(int(max(cleanyear)) + 5)
    #data = data[(data['year'] > minyear) & (data['year'] < maxyear)]
    columns = ['genres']
    # Combine text data in one column
    df_content = (pd.Series(data[columns]
                      .fillna('')
                      .values.tolist()).str.join(' '))
    # Instantiating and generating the count matrix
    count_vec = CountVectorizer()
    count_matrix = count_vec.fit_transform(df_content)
    # Get cosine similarity matrix
    cosine_sim = cosine_similarity(count_matrix, count_matrix)
    # Get the title of each movie as the index
    indices = pd.Series(data['title'])
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
    # Get top 50 indexes of our series as a list
    top_50_indexes = list(listings.iloc[:50].index)
    # Removing chosen movies
    top_indexes = np.setdiff1d(top_50_indexes,[idx_1,idx_2,idx_3])
    # Get top 10 movies that are most similar to the users chosen 3 movies
    for i in top_indexes[:top_n]:
        recommended_movies.append(list(movies['title'])[i])
    return recommended_movies
