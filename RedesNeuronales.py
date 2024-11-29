import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics.pairwise import cosine_similarity

# Función para cargar y procesar los datos
def load_data():
    # Cargar el dataset
    url = '___Top 100 IMDB Movies.csv'  # Cambia esto por la URL o ruta local de tu archivo CSV
    df = pd.read_csv(url, sep=';')

    # Crear una nueva columna que combine la descripción y el género
    df['content'] = df['description'] + ' ' + df['genre'].apply(lambda x: ' '.join(eval(x)))  # Convirtiendo lista de géneros en string

    return df

# Crear el modelo de red neuronal para generar representaciones de las películas
def create_model(vocab_size, max_len):
    model = Sequential([
        Dense(128, input_dim=X_train.shape[1], activation='relu'),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(64, activation='relu'),
        Dense(128, activation='relu'),
        Dense(X_train.shape[1], activation='sigmoid')  # Output layer
    ])

    return model

# Función para hacer la recomendación
def get_recommendations(movie_name, df, model, tokenizer, max_len):
    # Obtener el índice de la película seleccionada
    movie_index = df[df['title'] == movie_name].index[0]

    # Preprocesar el texto de la película seleccionada
    seq = tokenizer.texts_to_sequences([df['content'][movie_index]])
    seq = pad_sequences(seq, maxlen=max_len)

    # Generar la representación de la película seleccionada
    movie_vector = model.predict(seq)

    # Calcular la similitud con las otras películas
    all_movie_vectors = np.array([model.predict(pad_sequences(tokenizer.texts_to_sequences([text]), maxlen=max_len)) for text in df['content']])
    similarities = cosine_similarity(movie_vector, all_movie_vectors).flatten()

    # Obtener las películas más similares
    similar_indices = similarities.argsort()[-6:-1][::-1]  # Tomar las 5 más similares (sin incluir la misma película)
    
    recommended_movies = []
    for idx in similar_indices:
        recommended_movies.append({
            "title": df['title'][idx],
            "rating": df['rating'][idx],
            "year": df['year'][idx],
            "genre": df['genre'][idx],
            "image": df['image'][idx],
            "imdbid": df['imdbid'][idx]
        })

    return recommended_movies

# Configuración de la aplicación Streamlit
st.title("Sistema de Recomendación de Películas")
st.write("Bienvenido al sistema de recomendación de películas basado en aprendizaje profundo. Selecciona una película y obtén recomendaciones similares.")

# Cargar los datos
df = load_data()

# Tokenizer para preprocesar las descripciones
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['content'])
vocab_size = len(tokenizer.word_index) + 1
max_len = max(df['content'].apply(lambda x: len(x.split())))

# Crear el modelo
model = create_model(vocab_size, max_len)

# Entrenar el modelo (esto es solo un ejemplo, puedes ajustar los datos o el proceso de entrenamiento)
X_train = tokenizer.texts_to_sequences(df['content'])
X_train = pad_sequences(X_train, maxlen=max_len)

# Entrenamos el modelo en las descripciones de las películas
model.fit(X_train, X_train, epochs=5, batch_size=32, validation_split=0.1)

# Selección de película mediante un selector en Streamlit
movie_name = st.selectbox("Selecciona una película:", df['title'].tolist())

# Botón para obtener las recomendaciones
if st.button('Obtener recomendaciones'):
    st.write(f"Película seleccionada: {movie_name}")

    # Obtener las películas recomendadas
    recommended_movies = get_recommendations(movie_name, df, model, tokenizer, max_len)

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
            # Mostrar la imagen de la película
            st.image(movie["image"], width=100)

        st.write(f"[Ver en IMDb](https://www.imdb.com/title/{movie['imdbid']})")
