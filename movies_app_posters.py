# Importation des bibliothèques nécessaires
import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import requests
API_KEY = '831c97e561b1bea3a3188019a606068d'

def get_poster_url(title):
    url = f"https://api.themoviedb.org/3/movie/{title}?api_key={API_KEY}"
    response = requests.get(url)
    data = response.json()
    poster_path = data.get('poster_path')
    if poster_path:
        return f"https://image.tmdb.org/t/p/w500{poster_path}"
    return None

df = pd.read_csv("data.csv")

def update_poster_urls(df):
    df['poster_url'] = df['title'].apply(lambda x: get_poster_url(x))
    return df

df = update_poster_urls(df)

# Sélection des colonnes à utiliser pour les recommandations
columns_reco = df.columns[2:]  # Les colonnes de caractéristiques commencent à partir de la troisième colonne
X = df[columns_reco]

# Standardisation des données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Création et entraînement du modèle k-NN
knn_model = NearestNeighbors(n_neighbors=6, metric='cosine')
knn_model.fit(X_scaled)

# Fonction de recommandation
def recommend_movies(movie_title):
    try:
        movie_index = df[df["title"] == movie_title].index[0]
        _, indices = knn_model.kneighbors([X_scaled[movie_index]])

        recommended_movies_index = indices[0][1:]
        recommendations = df["title"].iloc[recommended_movies_index]
        return recommendations
    except IndexError:
        return pd.Series()  # Retourne une série vide si l'indice est introuvable

# Application Streamlit
def main():
    st.title("Système de recommandation de films")
    
    # Sélecteur de film
    movie_title = st.selectbox("Choisissez un film", df['title'])

    if st.button("Recommander"):
        recommendations = recommend_movies(movie_title)
        if not recommendations.empty:  # Vérifie si la série n'est pas vide
            st.write("Recommandations pour le film ", movie_title, ":")
            for title in recommendations:
                st.write(title)
        else:
            st.write("Aucune recommandation trouvée pour ce film.")

if __name__ == "__main__":
    main()
