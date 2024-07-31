# Importation des bibliothèques nécessaires
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

# Fonction pour charger les données de recommandation
@st.cache
def load_recommendation_data():
    return pd.read_csv("data1.csv")  # Remplacez par le chemin de votre fichier

# # Fonction pour charger les données d'analyse des acteurs
# @st.cache
# def load_analysis_data():
#     return pd.read_csv('movies_france_2000.csv', low_memory=False)

# Fonction de recommandation
def recommend_movies(movie_title, df, knn_model, X_scaled):
    try:
        movie_index = df[df["primaryTitle"] == movie_title].index[0]
        _, indices = knn_model.kneighbors([X_scaled[movie_index]])

        recommended_movies_index = indices[0][1:]
        recommendations = df["primaryTitle"].iloc[recommended_movies_index]
        return recommendations
    except IndexError:
        return pd.Series()  # Retourne une série vide si l'indice est introuvable
    
# Pour afficher les images
def get_poster_path(df, item_id):
    poster_path = df[df['tconst'] == item_id]['poster_path'].values[0]
    full_path = "https://image.tmdb.org/t/p/w500/" + poster_path
    return full_path



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
    movie_title = st.selectbox("Choisissez un film", df['primaryTitle'])

    if st.button("Recommander"):
        recommendations = recommend_movies(movie_title, df, knn_model, X_scaled)
        if not recommendations.empty:  # Vérifie si la série n'est pas vide
            st.write("Recommandations pour le film ", movie_title, ":")
            for title in recommendations:
                st.write(title)
                st.image(get_poster_path(df['tconst']), width=150)
        else:
            st.write("Aucune recommandation trouvée pour ce film.")

    # # Section d'analyse des acteurs
    # st.title("Analyse des Acteurs")

    # # Chargement des données d'analyse
    # df_analysis = load_analysis_data()

    # # Assurer que birthYear est numérique et supprimer les valeurs manquantes
    # df_analysis['birthYear'] = pd.to_numeric(df_analysis['birthYear'], errors='coerce')
    # df_analysis = df_analysis.dropna(subset=['birthYear'])

    # # Calculer l'âge des acteurs en supposant que l'année actuelle est 2024
    # df_analysis['age'] = 2024 - df_analysis['birthYear']

    # # Compter le nombre d'apparitions de chaque acteur
    # acteur_count = df_analysis['primaryName'].value_counts().reset_index()
    # acteur_count.columns = ['primaryName', 'nombre_apparitions']

    # # Identifier les 10 acteurs les plus actifs
    # top_10_acteurs = acteur_count.head(10)

    # # Joindre pour obtenir les âges des top 10 acteurs
    # top_10_df = df_analysis[df_analysis['primaryName'].isin(top_10_acteurs['primaryName'])]

    # # Calculer l'âge moyen de chaque acteur parmi les 10 plus actifs
    # age_moyen_top_10 = top_10_df.groupby('primaryName')['age'].mean().reset_index()

    # # Créer un graphique à barres pour représenter l'âge moyen des 10 acteurs les plus actifs
    # fig1, ax1 = plt.subplots(figsize=(12, 8))
    # sns.barplot(data=age_moyen_top_10, x='primaryName', y='age', order=top_10_acteurs['primaryName'], ax=ax1)
    # ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
    # ax1.set_xlabel('Acteur')
    # ax1.set_ylabel('Âge moyen')
    # ax1.set_title('Âge moyen des 10 acteurs les plus actifs')

    # # Afficher le graphique avec Streamlit
    # st.pyplot(fig1)

    #  # Chargement des données d'analyse
    # df = load_analysis_data()   

    # # Assurer que birthYear et deathYear sont numériques et supprimer les valeurs manquantes
    # df['birthYear'] = pd.to_numeric(df['birthYear'], errors='coerce')
    # df['deathYear'] = pd.to_numeric(df['deathYear'], errors='coerce')
    # df = df.dropna(subset=['birthYear'])

    # # Calculer l'âge des acteurs (utiliser la colonne deathYear si présente, sinon utiliser l'année actuelle 2024)
    # current_year = 2024
    # df['age'] = df.apply(lambda row: (row['deathYear'] if pd.notna(row['deathYear']) else current_year) - row['birthYear'], axis=1)

    # # Filtrer les lignes par profession (acteur/actrice)
    # genres = ['actor', 'actress']
    # df_actors = df[df['primaryProfession'].str.contains('|'.join(genres), na=False)]

    # # Créer des sous-ensembles pour les acteurs et actrices
    # df_male_actors = df_actors[df_actors['primaryProfession'].str.contains('actor', na=False) & ~df_actors['primaryProfession'].str.contains('actress', na=False)]
    # df_female_actors = df_actors[df_actors['primaryProfession'].str.contains('actress', na=False)]

    # # Créer un histogramme de la répartition de l'âge par genre
    # fig2, ax2 = plt.subplots(figsize=(12, 8))
    # bins = range(0, 110, 5)  # Définir des intervalles de 5 ans jusqu'à 110 ans

    # ax2.hist(df_male_actors['age'], bins=bins, alpha=0.5, label='Acteurs (Hommes)', color='blue', edgecolor='black')
    # ax2.hist(df_female_actors['age'], bins=bins, alpha=0.5, label='Actrices (Femmes)', color='red', edgecolor='black')

    # ax2.set_xlabel('Âge')
    # ax2.set_ylabel('Nombre d\'acteurs')
    # ax2.set_title('Répartition de l\'âge des acteurs en fonction du genre')
    # ax2.legend()
    # ax2.grid(axis='y', linestyle='--', alpha=0.7)
    # ax2.set_xticks(bins)

    # # Afficher le graphique avec Streamlit
    # st.pyplot(fig2)

    # # Chargement des données d'analyse
    # df_directors = load_analysis_data() 

    # # Filtrer les lignes où 'primaryProfession' contient 'director'
    # df_directors = df[df['primaryProfession'].str.contains('director', na=False)].copy()

    # # Assurer que averageRating et numVotes sont numériques et supprimer les valeurs manquantes
    # df_directors.loc[:, 'averageRating'] = pd.to_numeric(df_directors['averageRating'], errors='coerce')
    # df_directors.loc[:, 'numVotes'] = pd.to_numeric(df_directors['numVotes'], errors='coerce')
    # df_directors = df_directors.dropna(subset=['averageRating', 'numVotes'])

    # # Calculer le score moyen pondéré pour chaque réalisateur
    # C = df_directors['averageRating'].mean()
    # m = df_directors['numVotes'].quantile(0.75)
    # df_directors.loc[:, 'weighted_rating'] = (
    #     df_directors['numVotes'] / (df_directors['numVotes'] + m) * df_directors['averageRating'] +
    #     m / (df_directors['numVotes'] + m) * C
    # )

    # # Grouper par réalisateur et calculer la moyenne des scores pondérés
    # director_scores = df_directors.groupby('primaryName').agg({'weighted_rating': 'mean', 'numVotes': 'sum'}).reset_index()

    # # Identifier les 10 réalisateurs avec les scores pondés les plus élevés
    # top_10_weighted_directors = director_scores.nlargest(10, 'weighted_rating')

    # # Créer un graphique à barres pour les 10 meilleurs réalisateurs par score moyen pondéré
    # fig3, ax3 = plt.subplots(figsize=(12, 8))
    # ax3.barh(top_10_weighted_directors['primaryName'], top_10_weighted_directors['weighted_rating'], color='skyblue')
    # ax3.set_xlabel('Score moyen pondéré')
    # ax3.set_ylabel('Réalisateur')
    # ax3.set_title('Les 10 meilleurs réalisateurs par score moyen pondéré')
    # ax3.invert_yaxis()  # Inverser l'axe y pour avoir le réalisateur avec le score le plus élevé en haut

    # # Afficher le graphique avec Streamlit
    # st.pyplot(fig3)


    # # Chargement des données d'analyse
    # df_actors = load_analysis_data() 

    # # Filtrer les lignes où 'primaryProfession' contient 'actor' ou 'actress'
    # df_actors = df[df['primaryProfession'].str.contains('actor|actress', na=False)]

    # # Grouper par 'nconst' et 'primaryName', puis compter le nombre de films
    # acteur_count = df_actors.groupby(['nconst', 'primaryName']).size().reset_index(name='nombre_films')

    # # Trier par le nombre de films en ordre décroissant
    # acteur_count = acteur_count.sort_values(by='nombre_films', ascending=False)

    # # Sélectionner les 10 acteurs les plus actifs pour le visuel
    # top_10_acteurs = acteur_count.head(10)

    # # Créer un graphique à barres pour les 10 acteurs les plus actifs
    # fig4, ax4 = plt.subplots(figsize=(10, 6))
    # ax4.barh(top_10_acteurs['primaryName'], top_10_acteurs['nombre_films'], color='brown')
    # ax4.set_xlabel('Nombre de films')
    # ax4.set_ylabel('Acteurs')
    # ax4.set_title('Top 10 des acteurs les plus actifs')
    # ax4.invert_yaxis()  # Inverser l'axe Y pour avoir le plus actif en haut

    # # Afficher le graphique avec Streamlit
    # st.pyplot(fig4)

    # # Chargement des données d'analyse
    # df_directors = load_analysis_data() 
    # # Filtrer les réalisateurs
    # df_directors = df[df['primaryProfession'].str.contains('director', na=False)]

    # # Compter le nombre de films pour chaque réalisateur
    # director_count = df_directors['primaryName'].value_counts().reset_index()
    # director_count.columns = ['primaryName', 'nombre_films']

    # # Visualiser les résultats pour les 10 réalisateurs les plus prolifiques
    # top_10_directors = director_count.head(10)

    # # Afficher le tableau des résultats
    # # st.write('Top 10 des réalisateurs les plus prolifiques:')
    # # st.write(top_10_directors)

    # # Créer le graphique
    # fig5, ax5 = plt.subplots(figsize=(12, 8))
    # ax5.barh(top_10_directors['primaryName'], top_10_directors['nombre_films'], color='lightblue')
    # ax5.set_xlabel('Nombre de films')
    # ax5.set_ylabel('Réalisateur')
    # ax5.set_title('Nombre de films par les 10 réalisateurs les plus prolifiques')
    # ax5.invert_yaxis()  # Inverser l'axe y pour avoir le réalisateur avec le plus de films en haut

    # # Afficher le graphique dans Streamlit
    # st.pyplot(fig5)



    # # Chargement des données d'analyse
    # df_directors = load_analysis_data() 

    # # Filtrer les réalisateurs
    # df_directors = df[df['primaryProfession'].str.contains('director', na=False)].copy()

    # # Assurer que averageRating et numVotes sont numériques et supprimer les valeurs manquantes
    # df_directors['averageRating'] = pd.to_numeric(df_directors['averageRating'], errors='coerce')
    # df_directors['numVotes'] = pd.to_numeric(df_directors['numVotes'], errors='coerce')
    # df_directors = df_directors.dropna(subset=['averageRating', 'numVotes'])

    # # Calculer le score moyen pondéré pour chaque réalisateur
    # C = df_directors['averageRating'].mean()
    # m = df_directors['numVotes'].quantile(0.75)
    # df_directors['weighted_rating'] = (
    #     df_directors['numVotes'] / (df_directors['numVotes'] + m) * df_directors['averageRating'] +
    #     m / (df_directors['numVotes'] + m) * C
    # )

    # # Grouper par réalisateur et calculer la moyenne des scores pondérés
    # director_scores = df_directors.groupby('primaryName').agg({'weighted_rating': 'mean', 'numVotes': 'sum'}).reset_index()

    # # Identifier les 10 réalisateurs avec les scores pondérés les plus élevés
    # top_10_directors = director_scores.nlargest(10, 'weighted_rating')


    # # Créer le graphique
    # fig6, ax6 = plt.subplots(figsize=(12, 8))
    # ax6.barh(top_10_directors['primaryName'], top_10_directors['weighted_rating'], color='orange')
    # ax6.set_xlabel('Score moyen pondéré')
    # ax6.set_ylabel('Réalisateur')
    # ax6.set_title('Les 10 meilleurs réalisateurs par score moyen pondéré')
    # ax6.invert_yaxis()  # Inverser l'axe y pour avoir le réalisateur avec le score le plus élevé en haut

    # # Afficher le graphique dans Streamlit
    # st.pyplot(fig6)

    #     # Chargement des données d'analyse
    # df_actors = load_analysis_data() 
    # genres = load_analysis_data() 

    # # Filtrer les lignes par genre
    # genres = ['actor', 'actress']
    # df_actors = df[df['primaryProfession'].str.contains('|'.join(genres), na=False)]

    # # Compter le nombre d'acteurs et d'actrices
    # actor_count = df_actors['primaryProfession'].str.contains('actor', na=False) & ~df_actors['primaryProfession'].str.contains('actress', na=False)
    # actress_count = df_actors['primaryProfession'].str.contains('actress', na=False)

    # num_actors = actor_count.sum()
    # num_actresses = actress_count.sum()

    # # Créer un graphique à secteurs (camembert)
    # labels = ['Acteurs (Hommes)', 'Actrices (Femmes)']
    # sizes = [num_actors, num_actresses]
    # colors = ['blue', 'pink']
    # explode = (0.1, 0)  # "exploser" la première tranche (acteurs)

    # fig7, ax7 = plt.subplots(figsize=(8, 8))
    # ax7.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
    # ax7.set_title('Distribution de Genre des Acteurs et Actrices')
    # ax7.axis('equal')  # Assurer que le camembert est dessiné comme un cercle

    # # Afficher le graphique dans Streamlit
    # st.pyplot(fig7)

if __name__ == "__main__":
    main()
