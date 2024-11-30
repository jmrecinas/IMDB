import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity

# 1. Cargar el dataset
url = '___Top 100 IMDB Movies.csv'  # Asegúrate de tener el archivo cargado correctamente
df = pd.read_csv(url, sep=';')

# 2. Instanciar el TfidfVectorizer
tfidf = TfidfVectorizer(stop_words='english')

# 3. Crear una nueva columna que combine la descripción y el género
df['content'] = df['description'] + ' ' + df['genre'].apply(lambda x: ' '.join(eval(x)))  # Convirtiendo lista de géneros en string

# 4. Vectorizar la columna 'content' (descripción + géneros)
X = tfidf.fit_transform(df['content'])

# 5. Crear el modelo de KNN para filtrado basado en contenido
knn = NearestNeighbors(n_neighbors=5, metric='cosine')
knn.fit(X)

# 6. Crear la matriz de valoraciones para filtrado colaborativo
ratings_matrix = df.pivot_table(index='title', values='rating', aggfunc='mean')
ratings_matrix = ratings_matrix.fillna(0)

# 7. Calcular la similitud de las películas basada en las valoraciones (filtrado colaborativo)
cosine_sim = cosine_similarity(ratings_matrix)

# 8. Función para obtener recomendaciones basadas en contenido (KNN + TF-IDF)
def content_based_recommendations(movie_name, knn, X, df, top_n=5):
    movie_index = df[df['title'] == movie_name].index[0]
    distances, indices = knn.kneighbors(X[movie_index], n_neighbors=top_n+1)
    
    recommended_movies = []
    for idx in indices[0][1:]:
        recommended_movies.append({
            "title": df['title'][idx],
            "rating": df['rating'][idx],
            "year": df['year'][idx],
            "genre": df['genre'][idx],
            "image": df['image'][idx],
            "imdbid": df['imdbid'][idx],
            "imdb_link": df['imdb_link'][idx]
        })
    return recommended_movies

# 9. Función para obtener recomendaciones basadas en filtrado colaborativo (similitud de valoraciones)
def collaborative_filtering(movie_name, cosine_sim, df, top_n=5):
    idx = df[df['title'] == movie_name].index[0]
    similarity_scores = list(enumerate(cosine_sim[idx]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    similar_movies = similarity_scores[1:top_n+1]
    
    recommended_movies = []
    for i in similar_movies:
        movie_index = i[0]
        movie_info = {
            "title": df['title'].iloc[movie_index],
            "rating": df['rating'].iloc[movie_index],
            "year": df['year'].iloc[movie_index],
            "genre": df['genre'].iloc[movie_index],
            "image": df['image'].iloc[movie_index],
            "imdbid": df['imdbid'].iloc[movie_index],
            "imdb_link": df['imdb_link'].iloc[movie_index]
        }
        recommended_movies.append(movie_info)
    
    return recommended_movies

# 10. Función para combinar las recomendaciones de ambos sistemas (contenido y colaborativo)
def get_combined_recommendations(movie_name, df, knn, X, cosine_sim, top_n=5):
    content_recommendations = content_based_recommendations(movie_name, knn, X, df, top_n)
    collaborative_recommendations = collaborative_filtering(movie_name, cosine_sim, df, top_n)
    all_recommendations = {rec['title']: rec for rec in content_recommendations + collaborative_recommendations}
    return list(all_recommendations.values())

# 11. Interfaz de Streamlit
st.title("Sistema de Recomendación de Películas")

# 12. Selección de película
movie_name = st.selectbox("Selecciona una película:", df['title'].tolist())

# 13. Botón para obtener las recomendaciones
if st.button('Obtener recomendaciones'):
    st.write(f"Película seleccionada: {movie_name}")
    
    # Obtener las recomendaciones combinadas (basadas en contenido y colaborativo)
    recommended_movies = get_combined_recommendations(movie_name, df, knn, X, cosine_sim, top_n=5)
    
    st.write("### Películas recomendadas:")
    
    for movie in recommended_movies:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.subheader(movie["title"])
            st.write(f"**Rating:** {movie['rating']}")
            st.write(f"**Año:** {movie['year']}")
            st.write(f"**Género(s):** {movie['genre']}")
        with col2:
            st.image(movie["image"], width=100)
        
        st.write(f"[Ver en IMDb]({movie['imdb_link']})")
