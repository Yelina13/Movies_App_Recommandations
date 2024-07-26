# Importation des bibliothèques nécessaires
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

# Fonction principale de l'application Streamlit
def main():
    st.title("Système de recommandation de films")

    # Chargement des données de films pour les recommandations
    @st.cache_data
    def load_data():
        return pd.read_csv("data.csv")  # Remplacez par le chemin de votre fichier

    df = load_data()

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

    # Chargement des données pour l'analyse des acteurs et réalisateurs
    @st.cache_data
    def load_analysis_data():
        return pd.read_csv('movies_france_2000.csv', low_memory=False)

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

    # Assurer que birthYear et deathYear sont numériques et supprimer les valeurs manquantes
    df_analysis['birthYear'] = pd.to_numeric(df_analysis['birthYear'], errors='coerce')
    df_analysis['deathYear'] = pd.to_numeric(df_analysis['deathYear'], errors='coerce')
    df_analysis = df_analysis.dropna(subset=['birthYear'])

    # Calculer l'âge des acteurs (utiliser la colonne deathYear si présente, sinon utiliser l'année actuelle 2024)
    current_year = 2024
    df_analysis['age'] = df_analysis.apply(lambda row: (row['deathYear'] if pd.notna(row['deathYear']) else current_year) - row['birthYear'], axis=1)

    # Filtrer les lignes par profession (acteur/actrice)
    genres = ['actor', 'actress']
    df_actors = df_analysis[df_analysis['primaryProfession'].str.contains('|'.join(genres), na=False)]

    # Créer des sous-ensembles pour les acteurs et actrices
    df_male_actors = df_actors[df_actors['primaryProfession'].str.contains('actor', na=False) & ~df_actors['primaryProfession'].str.contains('actress', na=False)]
    df_female_actors = df_actors[df_actors['primaryProfession'].str.contains('actress', na=False)]

    # Créer un histogramme de la répartition de l'âge par genre
    fig2, ax2 = plt.subplots(figsize=(12, 8))
    bins = range(0, 110, 5)  # Définir des intervalles de 5 ans jusqu'à 110 ans

    ax2.hist(df_male_actors['age'], bins=bins, alpha=0.5, label='Acteurs (Hommes)', color='blue', edgecolor='black')
    ax2.hist(df_female_actors['age'], bins=bins, alpha=0.5, label='Actrices (Femmes)', color='red', edgecolor='black')

    ax2.set_xlabel('Âge')
    ax2.set_ylabel('Nombre d\'acteurs')
    ax2.set_title('Répartition de l\'âge des acteurs en fonction du genre')
    ax2.legend()
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    ax2.set_xticks(bins)

    # Afficher le graphique avec Streamlit
    st.pyplot(fig2)

    # Filtrer les lignes où 'primaryProfession' contient 'director'
    df_directors = df_analysis[df_analysis['primaryProfession'].str.contains('director', na=False)].copy()

    # Assurer que averageRating et numVotes sont numériques et supprimer les valeurs manquantes
    df_directors['averageRating'] = pd.to_numeric(df_directors['averageRating'], errors='coerce')
    df_directors['numVotes'] = pd.to_numeric(df_directors['numVotes'], errors='coerce')
    df_directors = df_directors.dropna(subset=['averageRating', 'numVotes'])

    # Calculer le score moyen pondéré pour chaque réalisateur
    C = df_directors['averageRating'].mean()
    m = df_directors['numVotes'].quantile(0.75)
    df_directors['weighted_rating'] = (
        df_directors['numVotes'] / (df_directors['numVotes'] + m) * df_directors['averageRating'] +
        m / (df_directors['numVotes'] + m) * C
    )

    # Grouper par réalisateur et calculer la moyenne des scores pondérés
    director_scores = df_directors.groupby('primaryName').agg({'weighted_rating': 'mean', 'numVotes': 'sum'}).reset_index()

    # Identifier les 10 réalisateurs avec les scores pondés les plus élevés
    top_10_weighted_directors = director_scores.nlargest(10, 'weighted_rating')

    # Créer un graphique à barres pour les 10 meilleurs réalisateurs par score moyen pondéré
    fig3, ax3 = plt.subplots(figsize=(12, 8))
    ax3.barh(top_10_weighted_directors['primaryName'], top_10_weighted_directors['weighted_rating'], color='skyblue')
    ax3.set_xlabel('Score moyen pondéré')
    ax3.set_ylabel('Réalisateur')
    ax3.set_title('Les 10 meilleurs réalisateurs par score moyen pondéré')
    ax3.invert_yaxis()  # Inverser l'axe y pour avoir le réalisateur avec le score le plus élevé en haut

    # Afficher le graphique avec Streamlit
    st.pyplot(fig3)




