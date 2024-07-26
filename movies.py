# Importation des bibliothèques nécessaires
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

# Fonction pour charger les données de recommandation
@st.cache_data
def load_recommendation_data():
    return pd.read_csv("data.csv")  # Remplacez par le chemin de votre fichier

# Fonction pour charger les données d'analyse des acteurs
@st.cache_data
def load_analysis_data():
    return pd.read_csv('movies_france_2000.csv', low_memory=False)

# Fonction de recommandation
def recommend_movies(movie_title, df, knn_model, X_scaled):
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
    # Section de recommandation
    st.title("Système de recommandation de films")

    # Chargement des données et préparation du modèle
    df = load_recommendation_data()
    columns_reco = df.columns[2:]  # Les colonnes de caractéristiques commencent à partir de la troisième colonne
    X = df[columns_reco]

    # Standardisation des données
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Création et entraînement du modèle k-NN
    knn_model = NearestNeighbors(n_neighbors=6, metric='cosine')
    knn_model.fit(X_scaled)

    # Sélecteur de film
    movie_title = st.selectbox("Choisissez un film", df['title'])

    if st.button("Recommander"):
        recommendations = recommend_movies(movie_title, df, knn_model, X_scaled)
        if not recommendations.empty:  # Vérifie si la série n'est pas vide
            st.write("Recommandations pour le film ", movie_title, ":")
            for title in recommendations:
                st.write(title)
        else:
            st.write("Aucune recommandation trouvée pour ce film.")

    # Section d'analyse des acteurs
    st.title("Analyse des Acteurs")

    # Chargement des données d'analyse
    df_analysis = load_analysis_data()

    # Assurer que birthYear est numérique et supprimer les valeurs manquantes
    df_analysis['birthYear'] = pd.to_numeric(df_analysis['birthYear'], errors='coerce')
    df_analysis = df_analysis.dropna(subset=['birthYear'])

    # Calculer l'âge des acteurs en supposant que l'année actuelle est 2024
    df_analysis['age'] = 2024 - df_analysis['birthYear']

    # Compter le nombre d'apparitions de chaque acteur
    acteur_count = df_analysis['primaryName'].value_counts().reset_index()
    acteur_count.columns = ['primaryName', 'nombre_apparitions']

    # Identifier les 10 acteurs les plus actifs
    top_10_acteurs = acteur_count.head(10)

    # Joindre pour obtenir les âges des top 10 acteurs
    top_10_df = df_analysis[df_analysis['primaryName'].isin(top_10_acteurs['primaryName'])]

    # Calculer l'âge moyen de chaque acteur parmi les 10 plus actifs
    age_moyen_top_10 = top_10_df.groupby('primaryName')['age'].mean().reset_index()

    # Créer un graphique à barres pour représenter l'âge moyen des 10 acteurs les plus actifs
    fig1, ax1 = plt.subplots(figsize=(12, 8))
    sns.barplot(data=age_moyen_top_10, x='primaryName', y='age', order=top_10_acteurs['primaryName'], ax=ax1)
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
    ax1.set_xlabel('Acteur')
    ax1.set_ylabel('Âge moyen')
    ax1.set_title('Âge moyen des 10 acteurs les plus actifs')

    # Afficher le graphique avec Streamlit
    st.pyplot(fig1)

if __name__ == "__main__":
    main()
