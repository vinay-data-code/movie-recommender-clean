import streamlit as st
import pandas as pd
import requests
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# API
import os
API_KEY = os.getenv("73a0a5b2c907006e9510b0468c80ff95")

# Load dataset
movies = pd.read_csv("tmdb_5000_movies.csv")
credits = pd.read_csv("tmdb_5000_credits.csv")

movies = movies.merge(credits, on="title")

movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]

# -------- FUNCTIONS -------- #

def convert(text):
    L = []
    for i in ast.literal_eval(text):
        L.append(i['name'])
    return L

def get_top_cast(text):
    L = []
    counter = 0
    for i in ast.literal_eval(text):
        if counter < 3:
            L.append(i['name'])
            counter += 1
        else:
            break
    return L

def fetch_director(text):
    L = []
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            L.append(i['name'])
    return L

# Apply
movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
movies['cast'] = movies['cast'].apply(get_top_cast)
movies['crew'] = movies['crew'].apply(fetch_director)

movies['overview'] = movies['overview'].fillna('').apply(lambda x: x.split())

movies['genres'] = movies['genres'].apply(lambda x: [i.replace(" ","") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x: [i.replace(" ","") for i in x])
movies['cast'] = movies['cast'].apply(lambda x: [i.replace(" ","") for i in x])
movies['crew'] = movies['crew'].apply(lambda x: [i.replace(" ","") for i in x])

movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

new_df = movies[['movie_id','title','tags']]
new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x).lower())

cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(new_df['tags']).toarray()

similarity = cosine_similarity(vectors)

# -------- API FUNCTIONS -------- #

def fetch_poster(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={API_KEY}"
    data = requests.get(url).json()
    return "https://image.tmdb.org/t/p/w500/" + data['poster_path']

def fetch_movie_details(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={API_KEY}"
    data = requests.get(url).json()
    return (
        "https://image.tmdb.org/t/p/w500/" + data['poster_path'],
        data['title'],
        data['vote_average'],
        data['overview']
    )

def fetch_cast(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}/credits?api_key={API_KEY}"
    data = requests.get(url).json()

    names = []
    images = []

    for i in data['cast'][:5]:
        names.append(i['name'])
        if i['profile_path']:
            images.append("https://image.tmdb.org/t/p/w500/" + i['profile_path'])
        else:
            images.append("https://via.placeholder.com/150")

    return names, images

def recommend(movie):
    index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[index]

    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    names = []
    posters = []

    for i in movies_list:
        movie_id = new_df.iloc[i[0]].movie_id
        names.append(new_df.iloc[i[0]].title)
        posters.append(fetch_poster(movie_id))

    return names, posters

# -------- UI -------- #

st.title("🎬 Movie Recommender System")

search = st.text_input("Search Movie")

selected_movie = None

if search:
    result = new_df[new_df['title'].str.contains(search, case=False)]
    if not result.empty:
        selected_movie = st.selectbox("Select Movie", result['title'])

if st.button("Recommend") and selected_movie:

    movie_id = new_df[new_df['title'] == selected_movie].iloc[0].movie_id

    poster, title, rating, overview = fetch_movie_details(movie_id)

    st.subheader("🎬 Selected Movie")

    col1, col2 = st.columns([1,2])
    with col1:
        st.image(poster)
    with col2:
        st.markdown(f"**{title}**")
        st.markdown(f"⭐ Rating: {rating}")
        st.write(overview)

    st.subheader("🎭 Cast")
    names, images = fetch_cast(movie_id)

    cols = st.columns(5)
    for i in range(5):
        with cols[i]:
            st.image(images[i])
            st.caption(names[i])

    st.subheader("🎯 Recommended")

    rec_names, rec_posters = recommend(selected_movie)

    cols = st.columns(5)
    for i in range(5):
        with cols[i]:
            st.image(rec_posters[i])
            st.caption(rec_names[i])
