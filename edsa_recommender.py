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
# Data for EDA
movies = pd.read_csv('resources/data/movies.csv')
ratings = pd.read_csv('resources/data/ratings.csv')
imdb = pd.read_csv('resources/data/imdb_data.csv')

# App declaration
def main():

    # DO NOT REMOVE the 'Recommender System' option below, however,
    # you are welcome to add more options to enrich your app.
    page_options = ["Home","Problem Statement", "Our Approach","Recommender System","Solution Overview", 
    "Exploratory Data Analysis", "App Developers"]

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

    if page_selection == "Home":
        #t.markdown(html_template.format('royalblue','white'), unsafe_allow_html=True)
        st.title("Movie Recommendation App")
        st.write("Team AE5 Unsupervised")
        
        st.image('resources/imgs/movies_selection.jpg',use_column_width=True) 

        st.header("Introduction")
        st.write("Recommendation systems are the systems that are used to gain more user attraction by understanding\
        the user’s preferences. These systems have now become popular because of their ability to provide personalized\
        content to different users to suit their taste. For example, millions of products listed on e-commerce websites\
        make it impossible to find out a desired product. This is where these systems help us by quickly recommending us\
        with the products we might be interested to buy. Another example is Netflix, which suggests the same genre movies\
        by understanding our interest or choice of movies we like. YouTube is another good example of recommender system,\
        which recommends videos using our historical data. There are many differenct recommendation engines available.\
        However, the scope of this project focuses on content based filtering and collaborative based filtering\
        recommendation systems.")

        st.subheader("So, what are you watching tonight...???")
        #st.markdown(title_template, unsafe_allow_html=True)

    if page_selection == "Problem Statement":
        st.title("Problem Statement")
        st.write("Create a recommendation algorithm based on content or collaborative filtering, capable of accurately\
        predicting how a user will rate a movie they have not yet viewed based on their historical preferences. We are also\
        tasked to expand on a template base streamlit app,\
        improving and fixing the given base recommender algorithms, as well as provide greater context to the problem.")
        st.write("For more information, please click here [link] (https://www.kaggle.com/c/edsa-movie-recommendation-challenge/overview).")
        
        st.image('resources/imgs/Movie_Recommendation.jpg',use_column_width=True)
    if page_selection == "Our Approach":
        st.title("Our approach")
        #st.write("Here we put a summary about how we solved the problem. *To added from our latest notebook*")
        st.write("The purpose of a recommendation system is to suggest relevant items to users. In our case these\
        items are movies. To achieve this task there are two main methods we can follow, namely Content Based Filtering\
        and Collaborative Based Filtering.")
        st.subheader("Collaborative Based Filtering")
        st.write("Recommender systems that use the collaborative approach rely solely on past interactions between recorded between users and\
        and items in order to produce new recommendations. These interactions are stored in a so-called 'utility matrix'.")
        st.write("The main idea that governs this approach is that past user-item interactions are sufficient to detect\
        similar users/items and to make predictions based on these estimated proximities.")
        st.write("Furthermore, this approach can be divided into two sub-categories which are known as memory based and\
        model based collaborative filtering. The memory based approach directly works with values of recorded interactions\
        ,assuming no model, and essentially based on the nearest neighbours search. The model based approach assume an\
        underlying model that explains the user-item interactions and try to discover it in order to make new predictions.\
        We used the model based approach to apply collaborative filtering.")
        st.write("The main advantage about collaborative approaches is that they require no informations about users or\
        items and so they can be used in many situations. Moreover, the more users interact with items the more new\
        recommendations becomes more accurate: for a fixed set of users and items, new interactions recorded over time\
        bring new information and make the system more effective.")
        st.write("However, as it only considers past interaction to make recommendations, collaborative filtering\
        suffer from the 'cold start problem': it is impossible to recommend anything to new users, or to recommend\
        a new item to users. This drawback can be addressed by recommending random items to new users or new items\
        to random users (random strategy), or by recommending popular items to new users or new items to the most active\
        users (maximum expectation strategy).")
        st.subheader("Content Based Filtering")
        st.write("Unlike collaborative methods that only rely on the user-item interactions, content based approaches\
        use additional information about users and/or items. If we consider the example of a movies recommender system,\
        this additional information can be, for example, the genre of the movie, the director, the title cast or the plot\
        keywords.")
        st.write("The idea of content based methods is to try to build a model, based on the available “features”,\
         that explain the observed user-item interactions. For example, what type of genre does a user like, does the\
        user have a favourate director, is the user a 'die-hard fan' of a specific actor/actress and what plots\
        intrigues the user are all features that can be considered when making recommendations.")
        st.write("Again, content based filtering can also suffer from the cold state problem unless we have user\
        based information like age and sex where we can recommend movies to a user based on these user features.\
        However, if we do not have user based information we can use the random strategy or maximum expectation strategy\
        to make recommendations to new users.")

    if page_selection == "Solution Overview":
        st.title("Solution Overview")
        #st.write("Describe your winning approach on this page")
        st.write("Recommendation Systems are very popular and useful for people to take proper automated decisions.\
            It is a method that helps a user to find out the information which is beneficial to him/her from a variety of data available.\
                The main goal of a recommendation system is to forecast the rating which a specific user gives to an item.\
                When it comes to Movie Recommendation Systems, recommendations are done based on the similarity between users (Collaborative Filtering)\
                    or by considering a particular user's activity based on what he wants to engage with (Content Based Filtering).")
        st.write('We will use both approaches to recommend movies to a user based on their 3 favourite movies.')
        st.subheader("Content based filtering")   
        st.markdown('- Applying the content based approach, we used the movies and imdb dataset.')
        st.markdown('- We then merged these two datasets in order the enrich the content available.')
        st.markdown('- The algorithm then recommended those movies which were most similar to the 3 favorite movies chosen by the user.')
        st.markdown('- The genre, title cast, director and plot keywords were taken into account when making these recommendations.')
        st.subheader("Collaborative based filtering")
        st.markdown('- The ratings and movies dataset was used to apply the collaborative based approach.')
        st.markdown('- The algorithm mapped the given favorite movies to users within the movies dataset.')
        st.markdown("- It then extracted the user ID's of users with similar high ratings for each movie.") 
        st.markdown('- Next, the top 10 movies were recommended for the user using the system.\
            These recommendations were made based on their similar high ratings by other similar users.')
        st.write('In order to fine tune these recommendations, a filter was applied in both approaches, filtering\
            out movies that was not in a specified release year range.')
        st.subheader("Performance of models investigated for Collaborative Filtering")
        st.write("RMSE of the recommendation models to show their performance")
        st.image('resources/imgs/RMSE.PNG',use_column_width=True)
        st.write("As we can see, the SVD model performed best, thus the SVD model was used in the collaborative filtering\
            approach.")
        st.subheader('Trade-offs')
        st.write('Both approaches gave relatively good recommendations, however the latency for the content based filter was lower\
            than that of the collaboratory based filter.')
        st.write("For the content based filter, if a user picks a list of movies which is not similar to any other movies in the dataset\
            based on genre, title cast, director or plot keywords then the recommendations given by the system won't be\
            very good.")    
        st.write("The same issue applies for the collaborative based filter, if no one else in the system has rated any\
             of the movies given by the user we won't be able to use this approach. However, there is a way around this\
                 problem by imputing the rating for this movie.")
        st.write("Another aspect we should consider is the diversity of recommendations. Collaborative based approaches \
            are known to give more diverse recommendations.\
            Nevertheless, the diversity of the recommendations for the content based approach can be improved by not\
                just determining similarity based on genre, but also adding plot keywords, directors and the title cast in\
                    the mix.")
        st.write("The computational requirements is also another big trade-off to consider. Both methods require computation of similarity structures. However,\
            the similarity structure for the collaborative based approach is larger. This can also seen given the lower\
                latency for the content based approach. Moreover, we are only using the ratings dataset which is\
                    only a subset of the full ratings data. If we used the full ratings data, our similarity structure\
                        would be even larger and hence require more computational power. Thus, if memory is an issue, it's\
            best suited to follow a content based approach.") 
        st.write("Furthermore, for the content based approach the similarity\
            matrix can be reused (unless more features are used to compute the similarity matrix). However, this is not\
            the case for the collaborative approach. As more users give ratings, the similarity matrix needs to be\
                updated in order to make improved recommendations.")
        st.subheader('So which one?')
        st.write("Well, it depends on what is more important to you. If you are looking for a recommendation system\
        that works faster then content based is the best option. If you want a system that suggests more diverse movies,\
        then the collaborative approach is the way to go. Both approaches can suffer from the cold start problem, but\
        the collaborative approach is more likely to suffer from this problem. However as time passes, more ratings are\
        given reducing the sparsity of the similarity structure which is very beneficial for the collaborative approach,\
        especially when working with large datasets. Then again, if you're working on your local computer and your computer is nothing special :),\
        then the content based approach will be better\
        suited.")
        st.subheader('Future work')
        st.write("We see that both of these methods has its pros and cons. In order to get the best of both worlds we can use\
        a hybrid approach which combines the content and collaborative based approaches.")
    # EDA
    st.cache(persist=True)
    if page_selection == "Exploratory Data Analysis":
        # EDA selection
        st.title('Explore Datasets')
        st.write('Choose dataset')
        sys = st.radio("Select an dataset",
                       ('movies',
                        'ratings','ratings_movies', 'imdb'))

        if sys == 'movies':
            st.title('EDA for movies dataset')
        
            
            if st.checkbox('Preview Movies Dataset'):
                if st.button('Head'):
                    st.write(movies.head())
                elif st.button('Tail'):
                    st.write(movies.tail())
            genres = pd.DataFrame(movies['genres'].
                      str.split("|").
                      tolist(),
                      index=movies['movieId']).stack()
            genres = genres.reset_index([0, 'movieId'])
            genres.columns = ['movieId', 'Genre']
            fig, ax = plt.subplots(figsize=(14, 7))
            sns.countplot(x='Genre',
              data=genres,
              palette='CMRmap',
              order=genres['Genre'].
              value_counts().index)
            plt.xticks(rotation=90)
            plt.xlabel('Genre', size=20)
            plt.ylabel('Count', size=20)
            plt.title('Distribution of Movie Genres', size=25)
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot()

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
            if st.checkbox('Show top 10 users that rated movies'):
                sns.barplot(y="total number of ratings", x="userId", data=user_id.head(10),
                order = user_id.head(10).sort_values('total number of ratings', ascending=False).userId, palette='magma')
                plt.xticks(rotation=45)
                plt.title('Top 10 users that rated movies')
                st.set_option('deprecation.showPyplotGlobalUse', False)
                st.pyplot()

        if sys == 'imdb':
            st.title('EDA for Imdb Dataset')
            if st.checkbox('Preview Imdb Dataset'):
                if st.button('Head'):
                    st.write(imdb.head())
                elif st.button('Tail'):
                    st.write(imdb.tail())
            if st.checkbox('Show top directors which appear the most'):
                def top_n_directors(df,column, n):
                    plt.figure(figsize=(14,7))
                    data = df[column].value_counts().head(n)
                    ax = sns.barplot(x = data.index, y = data, order= data.index, palette='CMRmap', edgecolor="black")
                    for p in ax.patches:
                        ax.text(p.get_x() + p.get_width()/2., p.get_height(), '%d' % int(p.get_height()), fontsize=11, ha='center', va='bottom')
                    plt.title(f'Top {n} {column.title()}', fontsize=14)
                    plt.xlabel(column.title())
                    plt.ylabel('Number of occurences')
                    plt.xticks(rotation=90)
                    plt.show()
                    st.set_option('deprecation.showPyplotGlobalUse', False)
                    st.pyplot()
                top_n_directors(imdb,'director',15)
            option = st.selectbox("Choose you favourite director", pd.unique(imdb['director']))
            st.write("You selected:",option)
            if st.checkbox("Show your favourite director's movies"):
                directors = imdb[imdb['director']==option]
                st.write(directors.head())

    if page_selection == "App Developers":
        st.title("App Developers")
        st.subheader("Team AE5 members:")
        st.write("1. Xichavo Hobyani")
        st.write("2. Ignitious Chauke")
        st.write("3. Boitumelo Makgoba")
        st.write("4. Muziwandile Mavundla")
        st.write("5. Hendri Kouter")
        st.write("6. Siphamandla Mandindi")
    # You may want to add more sections here for aspects such as an EDA,
    # or to provide your business pitch.


if __name__ == '__main__':
    main()
