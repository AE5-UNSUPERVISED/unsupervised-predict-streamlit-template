"""

    Streamlit webserver-based Recommender Engine.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within the root of this repository for guidance on how to use
    this script correctly.

    NB: !! Do not remove/modify the code delimited by dashes !!

    This application is intended to be partly marked in an automated manner.
    Altering delimited code may result in a mark of 0.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend certain aspects of this script
    and its dependencies as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st

# Data handling dependencies
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Custom Libraries
from utils.data_loader import load_movie_titles
from recommenders.collaborative_based import collab_model
from recommenders.content_based import content_model

# Data Loading
title_list = load_movie_titles('resources/data/movies.csv')
movies = pd.read_csv('resources/data/movies.csv')
ratings = pd.read_csv('resources/data/ratings.csv')

# App declaration
def main():

    # DO NOT REMOVE the 'Recommender System' option below, however,
    # you are welcome to add more options to enrich your app.
    page_options = ["Recommender System","Solution Overview", "EDA"]

    # -------------------------------------------------------------------
    # ----------- !! THIS CODE MUST NOT BE ALTERED !! -------------------
    # -------------------------------------------------------------------
    page_selection = st.sidebar.selectbox("Choose Option", page_options)
    if page_selection == "Recommender System":
        # Header contents
        st.write('# Movie Recommender Engine')
        st.write('### EXPLORE Data Science Academy Unsupervised Predict')
        st.image('resources/imgs/Image_header.png',use_column_width=True)
        # Recommender System algorithm selection
        sys = st.radio("Select an algorithm",
                       ('Content Based Filtering',
                        'Collaborative Based Filtering'))

        # User-based preferences
        st.write('### Enter Your Three Favorite Movies')
        movie_1 = st.selectbox('Fisrt Option',title_list[14930:15200])
        movie_2 = st.selectbox('Second Option',title_list[25055:25255])
        movie_3 = st.selectbox('Third Option',title_list[21100:21200])
        fav_movies = [movie_1,movie_2,movie_3]

        # Perform top-10 movie recommendation generation
        if sys == 'Content Based Filtering':
            if st.button("Recommend"):
                try:
                    with st.spinner('Crunching the numbers...'):
                        top_recommendations = content_model(movie_list=fav_movies,
                                                            top_n=10)
                    st.title("We think you'll like:")
                    for i,j in enumerate(top_recommendations):
                        st.subheader(str(i+1)+'. '+j)
                except:
                    st.error("Oops! Looks like this algorithm does't work.\
                              We'll need to fix it!")


        if sys == 'Collaborative Based Filtering':
            if st.button("Recommend"):
                try:
                    with st.spinner('Crunching the numbers...'):
                        top_recommendations = collab_model(movie_list=fav_movies,
                                                           top_n=10)
                    st.title("We think you'll like:")
                    for i,j in enumerate(top_recommendations):
                        st.subheader(str(i+1)+'. '+j)
                except:
                    st.error("Oops! Looks like this algorithm does't work.\
                              We'll need to fix it!")


    # -------------------------------------------------------------------

    # ------------- SAFE FOR ALTERING/EXTENSION -------------------
    if page_selection == "Solution Overview":
        st.title("Solution Overview")
        st.write("Describe your winning approach on this page")
    
    # EDA
    st.cache(persist=True)
    if page_selection == "EDA":
        # EDA selection
        st.title('Explore Datasets')
        st.write('Choose dataset')
        sys = st.radio("Select an dataset",
                       ('movies',
                        'ratings','ratings_movies'))

        if sys == 'movies':
            st.title('EDA for movies dataset')
        
            
            if st.checkbox('Preview Movies Dataset'):
                if st.button('Head'):
                    st.write(movies.head())
                elif st.button('Tail'):
                    st.write(movies.tail())

        if sys == 'ratings':
            st.title('EDA for ratings Dataset')
            if st.checkbox('Preview Ratings Dataset'):
                if st.button('Head'):
                    st.write(ratings.head())
                elif st.button('Tail'):
                    st.write(ratings.tail())

            # show distribution of ratings
            if st.checkbox('Show distribution of ratings'):
                sns.countplot(x = 'rating', data = ratings, palette="mako")
                plt.title('Distribution of ratings')
                st.set_option('deprecation.showPyplotGlobalUse', False)
                st.pyplot()

        if sys == 'ratings_movies':
            st.title('EDA for Ratings-Movies Dataset')
            # merge ratings and movies dataset
            ratings_movies = pd.merge(ratings, movies, on='movieId')

            # group ratings_movies by title and calc mean rating
            trend = pd.DataFrame(ratings_movies.groupby('title')['rating'].mean())
            # create new column by grouping by title counting the number of ratings per movie
            trend['total number of ratings'] = pd.DataFrame(ratings_movies.groupby('title')['rating'].count())    

            # sort dataframe by total number of ratings
            trend.sort_values(by=['total number of ratings'], inplace=True, ascending=False)
            # reset the index
            trend.reset_index(inplace=True)

            # show top 5 rated movies
            if st.checkbox('Show top 5 rated movies'):
                fig, ax = plt.subplots()
                sns.barplot(x="total number of ratings", y="title", data=trend.head(), palette='rocket')
                plt.title('Top 5 rated movies')
                st.pyplot(fig)

            # group ratings_movies by users
            user_id = pd.DataFrame(ratings_movies.groupby('userId')['rating'].mean())
            user_id['total number of ratings'] = pd.DataFrame(ratings_movies.groupby('userId')['rating'].count())
            # sort dataframe by total number of ratings
            user_id.sort_values(by=['total number of ratings'], inplace=True, ascending=False)
            # reset the index
            user_id.reset_index(inplace=True)

            # show movie fundi's
            if st.checkbox('Show top 10 movie fundis'):
                sns.barplot(y="total number of ratings", x="userId", data=user_id.head(10),
                order = user_id.head(10).sort_values('total number of ratings', ascending=False).userId, palette='magma')
                plt.xticks(rotation=45)
                plt.title('Top 10 movie fundis')
                st.set_option('deprecation.showPyplotGlobalUse', False)
                st.pyplot()
        
    # You may want to add more sections here for aspects such as an EDA,
    # or to provide your business pitch.


if __name__ == '__main__':
    main()
