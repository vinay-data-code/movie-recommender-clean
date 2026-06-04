import streamlit as st
import pickle
import pandas as pd
import requests

# Load data
movies_dict = pickle.load(open('movies.pkl','rb'))
movies = pd.DataFrame(movies_dict)

similarity = pickle.load(open('similarity.pkl','rb'))

API_KEY = "73a0a5b2c907006e9510b0468c80ff95"

# -------- FUNCTIONS -------- #

def fetch_poster(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={API_KEY}"
    data = requests.get(url).json()
    return "https://image.tmdb.org/t/p/w500/" + data['poster_path']

def fetch_cast(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}/credits?api_key={API_KEY}"
    data = requests.get(url).json()
    
    cast_names = []
    cast_images = []
    
    for i in data['cast'][:5]:
        cast_names.append(i['name'])
        
        if i['profile_path']:
            cast_images.append("https://image.tmdb.org/t/p/w500/" + i['profile_path'])
        else:
            cast_images.append("https://via.placeholder.com/150")

    return cast_names, cast_images

def recommend(movie):
    index = movies[movies['title'] == movie].index[0]
    distances = similarity[index]
    
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    recommended_movies = []
    recommended_posters = []

    for i in movies_list:
        movie_id = movies.iloc[i[0]].movie_id

        recommended_movies.append(movies.iloc[i[0]].title)
        recommended_posters.append(fetch_poster(movie_id))

    return recommended_movies, recommended_posters

def fetch_movie_details(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={API_KEY}"
    data = requests.get(url).json()
    
    poster = "https://image.tmdb.org/t/p/w500/" + data['poster_path']
    title = data['title']
    rating = data['vote_average']
    overview = data['overview']
    
    return poster, title, rating, overview


# -------- UI -------- #

st.title("🎬 Movie Recommender System")

search_term = st.text_input("🔍 Search movie")

selected_movie = None

if search_term:
    results = movies[movies['title'].str.contains(search_term, case=False, na=False)]
    
    if not results.empty:
        selected_movie = st.selectbox("Select movie", results['title'].values)
    else:
        st.warning("No movie found")


# ✅ BUTTON CLICK KE ANDAR SAB AAYEGA
if st.button('Recommend') and selected_movie:

    # movie id
    movie_id = movies[movies['title'] == selected_movie].iloc[0].movie_id

    # -------- SELECTED MOVIE -------- #
    poster, title, rating, overview = fetch_movie_details(movie_id)

    st.subheader("🎬 Selected Movie")
    col1, col2 = st.columns([1,2])

    with col1:
        st.image(poster)

    with col2:
        st.markdown(f"**{title}**")
        st.markdown(f"⭐ Rating: {rating}")
        st.markdown(f"{overview}")

    # -------- CAST -------- #
    st.subheader("🎭 Cast")

    cast_names, cast_images = fetch_cast(movie_id)

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.image(cast_images[0])
        st.caption(cast_names[0])

    with col2:
        st.image(cast_images[1])
        st.caption(cast_names[1])

    with col3:
        st.image(cast_images[2])
        st.caption(cast_names[2])

    with col4:
        st.image(cast_images[3])
        st.caption(cast_names[3])

    with col5:
        st.image(cast_images[4])
        st.caption(cast_names[4])

    # -------- RECOMMENDATIONS -------- #
    st.subheader("🎯 Recommended Movies")

    names, posters = recommend(selected_movie)

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.image(posters[0])
        st.caption(names[0])

    with col2:
        st.image(posters[1])
        st.caption(names[1])

    with col3:
        st.image(posters[2])
        st.caption(names[2])

    with col4:
        st.image(posters[3])
        st.caption(names[3])

    with col5:
        st.image(posters[4])
        st.caption(names[4])