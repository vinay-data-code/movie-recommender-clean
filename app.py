import streamlit as st
import pandas as pd
import requests
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os


# ✅ API KEY FIX
API_KEY = os.getenv("API_KEY")

# ✅ LOAD DATA FAST
@st.cache_data
def load_data():
    movies = pd.read_csv("tmdb_5000_movies.csv")
    credits = pd.read_csv("tmdb_5000_credits.csv")

    movies = movies.merge(credits, on="title")
    movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]

    def convert(text):
        return [i['name'] for i in ast.literal_eval(text)]

    def get_top_cast(text):
        return [i['name'] for i in ast.literal_eval(text)[:3]]

    def fetch_director(text):
        for i in ast.literal_eval(text):
            if i['job'] == 'Director':
                return [i['name']]
        return []

    movies['genres'] = movies['genres'].apply(convert)
    movies['keywords'] = movies['keywords'].apply(convert)
    movies['cast'] = movies['cast'].apply(get_top_cast)
    movies['crew'] = movies['crew'].apply(fetch_director)

    movies['overview'] = movies['overview'].fillna('').apply(lambda x: x.split())

    # ✅ SAFE TYPE FIX
    movies['genres'] = movies['genres'].apply(lambda x: x if isinstance(x, list) else [])
    movies['keywords'] = movies['keywords'].apply(lambda x: x if isinstance(x, list) else [])
    movies['cast'] = movies['cast'].apply(lambda x: x if isinstance(x, list) else [])
    movies['crew'] = movies['crew'].apply(lambda x: x if isinstance(x, list) else [])

    movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

    new_df = movies[['movie_id','title','tags']]
    new_df['tags'] = new_df['tags'].apply(lambda x: " ".join([str(i) for i in x]).lower())

    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(new_df['tags']).toarray()
    similarity = cosine_similarity(vectors)

    return new_df, similarity

new_df, similarity = load_data()

# ✅ FIX: MOVIE NAME → TMDB ID
def get_movie_id(movie_name):
    url = f"https://api.themoviedb.org/3/search/movie?api_key={API_KEY}&query={movie_name}"
    data = requests.get(url).json()

    if data.get('results'):
        return data['results'][0]['id']
    return None

# ✅ POSTER
def fetch_poster(movie_name):
    movie_id = get_movie_id(movie_name)
    if not movie_id:
        return "https://via.placeholder.com/300x450"

    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={API_KEY}"
    data = requests.get(url).json()

    if data.get('poster_path'):
        return "https://image.tmdb.org/t/p/w500/" + data['poster_path']
    return "https://via.placeholder.com/300x450"

# ✅ MOVIE DETAILS
def fetch_movie_details(movie_name):
    movie_id = get_movie_id(movie_name)
    if not movie_id:
        return ("https://via.placeholder.com/300x450","No Title","N/A","No description")

    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={API_KEY}"
    data = requests.get(url).json()

    poster = "https://image.tmdb.org/t/p/w500/" + data['poster_path'] if data.get('poster_path') else "https://via.placeholder.com/300x450"

    return poster, data.get('title','No Title'), data.get('vote_average','N/A'), data.get('overview','No description')

# ✅ CAST
def fetch_cast(movie_name):
    movie_id = get_movie_id(movie_name)

    if not movie_id:
        return ["No Actor"]*5, ["https://via.placeholder.com/150"]*5

    url = f"https://api.themoviedb.org/3/movie/{movie_id}/credits?api_key={API_KEY}"
    data = requests.get(url).json()

    if 'cast' not in data:
        return ["No Actor"]*5, ["https://via.placeholder.com/150"]*5

    names, images = [], []

    for i in data['cast'][:5]:
        names.append(i.get('name','No Actor'))
        if i.get('profile_path'):
            images.append("https://image.tmdb.org/t/p/w500/" + i['profile_path'])
        else:
            images.append("https://via.placeholder.com/150")

    return names, images

# ✅ RECOMMENDATION
def recommend(movie):
    index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[index]

    movies_list = sorted(list(enumerate(distances)),
                         reverse=True,
                         key=lambda x: x[1])[1:6]

    names, posters = [], []

    for i in movies_list:
        title = new_df.iloc[i[0]].title
        names.append(title)
        posters.append(fetch_poster(title))

    return names, posters

# ✅ UI
st.title("🎬 Movie Recommender System")

search = st.text_input("🔍 Search Movie")

selected_movie = None
if search:
    result = new_df[new_df['title'].str.contains(search, case=False)]
    if not result.empty:
        selected_movie = st.selectbox("Select Movie", result['title'])

if st.button("Recommend") and selected_movie:

    poster, title, rating, overview = fetch_movie_details(selected_movie)

    st.subheader("🎬 Selected Movie")
    col1, col2 = st.columns([1,2])

    with col1:
        st.image(poster)

    with col2:
        st.markdown(f"**{title}**")
        st.markdown(f"⭐ Rating: {rating}")
        st.write(overview)

    st.subheader("🎭 Cast")
    names, images = fetch_cast(selected_movie)

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