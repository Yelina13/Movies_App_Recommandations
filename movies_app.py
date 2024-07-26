# Importation des bibliothèques nécessaires
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

# Fonction pour charger les données
@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path, low_memory=False)

# Chargement des données de films pour les recommandations
df_recommendations = load_data("data.csv")

# Sélection des colonnes à utiliser pour les recommandations
columns_reco = df_recommendations.columns[2:]  # Les colonnes de caractéristiques commencent à partir de la troisième colonne
X = df_recommendations[columns_reco]

# Standardisation des données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Création et entraînement du modèle k-NN
knn_model = NearestNeighbors(n_neighbors=6, metric='cosine')
knn_model.fit(X_scaled)

# Fonction de recommandation
def recommend_movies(movie_title):
    try:
        movie_index = df_recommendations[df_recommendations["title"] == movie_title].index[0]
        _, indices = knn_model.kneighbors([X_scaled[movie_index]])

        recommended_movies_index = indices[0][1:]
        recommendations = df_recommendations["title"].iloc[recommended_movies_index]
        return recommendations
    except IndexError:
        return pd.Series()  # Retourne une série vide si l'indice est introuvable

# Application Streamlit
def main():
    st.title("Système de recommandation de films")
    
    # Sélecteur de film
    movie_title = st.selectbox("Choisissez un film", df_recommendations['title'])

    if st.button("Recommander"):
        recommendations = recommend_movies(movie_title)
        if not recommendations.empty:  # Vérifie si la série n'est pas vide
            st.write("Recommandations pour le film ", movie_title, ":")
            for title in recommendations:
                st.write(title)
        else:
            st.write("Aucune recommandation trouvée pour ce film.")

    # Chargement des données d'analyse
    df_analysis = load_data('movies_france_2000.csv')

    # Assurer que birthYear est numérique et supprimer les valeurs manquantes
    df_analysis['birthYear'] = pd.to_numeric(df_analysis['birthYear'], errors='coerce')
    df_analysis = df_analysis.dropna(subset=['birthYear'])

    # Calculer l'âge des acteurs en supposant que l'année actuelle est 2024
    df_analysis['age'] = 2024 - df_analysis['birthYear']

    # Analyse et visualisations
    top_10_acteurs = analyse_acteurs(df_analysis)
    visualiser_acteurs_age_moyen(df_analysis, top_10_acteurs)
    visualiser_age_repartition(df_analysis)
    visualiser_realisateurs_pondere(df_analysis)

# Fonctions d'analyse
def analyse_acteurs(df):
    # Compter le nombre d'apparitions de chaque acteur
    acteur_count = df['primaryName'].value_counts().reset_index()
    acteur_count.columns = ['primaryName', 'nombre_apparitions']

    # Identifier les 10 acteurs les plus actifs
    top_10_acteurs = acteur_count.head(10)
    return top_10_acteurs

def visualiser_acteurs_age_moyen(df, top_10_acteurs):
    # Joindre pour obtenir les âges des top 10 acteurs
    top_10_df = df[df['primaryName'].isin(top_10_acteurs['primaryName'])]

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

def visualiser_age_repartition(df):
    # Créer un histogramme de la répartition de l'âge par genre
    fig2, ax2 = plt.subplots(figsize=(12, 8))
    bins = range(0, 110, 5)  # Définir des intervalles de 5 ans jusqu'à 110 ans

    # Filtrer les lignes par profession (acteur/actrice)
    genres = ['actor', 'actress']
    df_actors = df[df['primaryProfession'].str.contains('|'.join(genres), na=False)]

    # Créer des sous-ensembles pour les acteurs et actrices
    df_male_actors = df_actors[df_actors['primaryProfession'].str.contains('actor', na=False) & ~df_actors['primaryProfession'].str.contains('actress', na=False)]
    df_female_actors = df_actors[df_actors['primaryProfession'].str.contains('actress', na=False)]

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

def visualiser_realisateurs_pondere(df):
    # Calculer le score moyen pondéré pour chaque réalisateur
    C = df['averageRating'].mean()
    m = df['numVotes'].quantile(0.75)
    df['weighted_rating'] = (
        df['numVotes'] / (df['numVotes'] + m) * df['averageRating'] +
        m / (df['numVotes'] + m) * C
    )

    # Grouper par réalisateur et calculer la moyenne des scores pondérés
    director_scores = df.groupby('primaryName').agg({'weighted_rating': 'mean', 'numVotes': 'sum'}).reset_index()

    # Identifier les 10 réalisateurs avec les scores pondérés les plus élevés
    top_10_directors = director_scores.nlargest(10, 'weighted_rating')

    # Créer un graphique à barres pour les 10 meilleurs réalisateurs par score moyen pondéré
    fig3, ax3 = plt.subplots(figsize=(12, 8))
    ax3.barh(top_10_directors['primaryName'], top_10_directors['weighted_rating'], color='orange')
    ax3.set_xlabel('Score moyen pondéré')
    ax3.set_ylabel('Réalisateur')
    ax3.set_title('Les 10 meilleurs réalisateurs par score moyen pondéré')
    ax3.invert_yaxis()  # Inverser l'axe y pour avoir le réalisateur avec le score le plus élevé en haut

    # Afficher le graphique avec Streamlit
    st.pyplot(fig3)

if __name__ == "__main__":
    main()

