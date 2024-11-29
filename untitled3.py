# -*- coding: utf-8 -*-
"""Untitled3.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1zaeuTW82OZMqR5tyLHS4H2O3h3he11mB
"""
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# Función para cargar y procesar los datos
def load_data():
    # Cargar el dataset
    url = '___Top 100 IMDB Movies.csv'  # Cambia esto por la URL o ruta local de tu archivo CSV
    df = pd.read_csv(url, sep=';')

    # Crear una nueva columna que combine la descripción y el género
    df['content'] = df['description'] + ' ' + df['genre'].apply(lambda x: ' '.join(eval(x)))  # Convirtiendo lista de géneros en string

    # Instanciar el TfidfVectorizer
    tfidf = TfidfVectorizer(stop_words='english')
    X = tfidf.fit_transform(df['content'])

    # Crear el modelo de KNN
    knn = NearestNeighbors(n_neighbors=5, metric='cosine')
    knn.fit(X)

    return df, knn, tfidf

# Función para hacer la recomendación
def get_recommendations(movie_name, df, knn, tfidf):
    # Obtener el índice de la película seleccionada
    movie_index = df[df['title'] == movie_name].index[0]

    # Buscar las 5 películas más cercanas usando KNN
    distances, indices = knn.kneighbors(tfidf.transform([df['content'][movie_index]]), n_neighbors=6)

    recommended_movies = []
    for idx in indices[0][1:]:
        recommended_movies.append({
            "title": df['title'][idx],
            "rating": df['rating'][idx],
            "year": df['year'][idx],
            "genre": df['genre'][idx],
            "imdb_link": df['imdb_link'][idx]
        })

    return recommended_movies

# Configuración de la aplicación Streamlit
st.title("Sistema de Recomendación de Películas")
st.write("Bienvenido al sistema de recomendación de películas basado en contenido. Selecciona una película y obtén recomendaciones similares.")

# Cargar los datos y modelo
df, knn, tfidf = load_data()

# Selección de película mediante un selector en Streamlit
movie_name = st.selectbox("Selecciona una película:", df['title'].tolist())

# Botón para obtener las recomendaciones
if st.button('Obtener recomendaciones'):
    st.write(f"Película seleccionada: {movie_name}")

    # Obtener las películas recomendadas
    recommended_movies = get_recommendations(movie_name, df, knn, tfidf)

    # Mostrar las recomendaciones en formato presentable
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

        st.write(f"[Ver en IMDb](https://www.imdb.com/title/{movie['imdb_link'][7:]})")
