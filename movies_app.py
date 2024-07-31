# Importation des bibliothèques nécessaires
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import logging
import random

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Configuration du style global pour les graphiques
sns.set(style="white", palette="pastel", font_scale=1.2)

# Fonction pour charger les données de recommandation
@st.cache  # Utiliser @st.cache_data pour les versions récentes de Streamlit
def load_recommendation_data() -> pd.DataFrame:
    logging.debug("Loading recommendation data...")
    return pd.read_csv("data.csv")  # Remplacez par le chemin de votre fichier

# Fonction pour charger les données d'analyse des acteurs
@st.cache
def load_analysis_data() -> pd.DataFrame:
    return pd.read_csv('movies_france_2000.csv', low_memory=False)

# Fonction de recommandation
def recommend_movies(movie_title: str, df: pd.DataFrame, knn_model: NearestNeighbors, X_scaled) -> pd.DataFrame:
    try:
        if movie_title not in df["title"].values:
            logging.warning(f"Movie title '{movie_title}' not found in the dataset.")
            return pd.DataFrame()  # Retourne un DataFrame vide si le film n'est pas trouvé

        movie_index = df[df["title"] == movie_title].index[0]
        _, indices = knn_model.kneighbors([X_scaled[movie_index]])

        recommended_movies_index = indices[0][1:]
        recommendations = df.iloc[recommended_movies_index]
        logging.info(f"Recommendations for '{movie_title}': {recommendations['title'].tolist()}")
        return recommendations
    except IndexError:
        logging.error(f"Index error for movie '{movie_title}'.")
        return pd.DataFrame()  # Retourne un DataFrame vide si l'indice est introuvable

# Fonction pour obtenir le chemin de l'affiche
def get_poster_path(df: pd.DataFrame, item_id: str) -> str:
    try:
        poster_path = df[df['tconst'] == item_id]['poster_path'].values[0]
        full_path = "https://image.tmdb.org/t/p/w500/" + poster_path
        logging.debug(f"Poster path for item '{item_id}': {full_path}")
        return full_path
    except IndexError:
        logging.error(f"Poster path not found for item '{item_id}'.")
        return ""

def create_graphs(df_analysis: pd.DataFrame):
    # Assurer que birthYear est numérique et supprimer les valeurs manquantes
    df_analysis['birthYear'] = pd.to_numeric(df_analysis['birthYear'], errors='coerce')
    df_analysis = df_analysis.dropna(subset=['birthYear'])

    # Calculer l'âge des acteurs en supposant que l'année actuelle est 2024
    df_analysis['age'] = 2024 - df_analysis['birthYear']

    # 1. Âge moyen des 10 acteurs les plus actifs
    acteur_count = df_analysis['primaryName'].value_counts().reset_index()
    acteur_count.columns = ['primaryName', 'nombre_apparitions']
    top_10_acteurs = acteur_count.head(10)
    top_10_df = df_analysis[df_analysis['primaryName'].isin(top_10_acteurs['primaryName'])]
    age_moyen_top_10 = top_10_df.groupby('primaryName')['age'].mean().reset_index()
    fig1, ax1 = plt.subplots(figsize=(12, 8))
    sns.barplot(data=age_moyen_top_10, x='primaryName', y='age', order=top_10_acteurs['primaryName'], ax=ax1)
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
    ax1.set_xlabel('Acteur', fontsize=18, fontweight='bold', family='serif')
    ax1.set_ylabel('Âge moyen', fontsize=14, fontweight='bold')
    ax1.set_title('Âge moyen des 10 acteurs les plus actifs', fontsize=14, fontweight='bold')

    # 2. Répartition de l'âge des acteurs en fonction du genre
    genres = ['actor', 'actress']
    df_actors = df_analysis[df_analysis['primaryProfession'].str.contains('|'.join(genres), na=False)]
    df_male_actors = df_actors[df_actors['primaryProfession'].str.contains('actor', na=False) & ~df_actors['primaryProfession'].str.contains('actress', na=False)]
    df_female_actors = df_actors[df_actors['primaryProfession'].str.contains('actress', na=False)]
    fig2, ax2 = plt.subplots(figsize=(12, 8))
    bins = range(0, 110, 5)
    ax2.hist(df_male_actors['age'], bins=bins, alpha=0.5, label='Acteurs (Hommes)', color='blue', edgecolor='black')
    ax2.hist(df_female_actors['age'], bins=bins, alpha=0.5, label='Actrices (Femmes)', color='red', edgecolor='black')
    ax2.set_xlabel('Âge', fontsize=18, fontweight='bold', family='serif')
    ax2.set_ylabel('Nombre d\'acteurs', fontsize=14, fontweight='bold')
    ax2.set_title('Répartition de l\'âge des acteurs en fonction du genre', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    ax2.set_xticks(bins)

    # 3. Les 10 meilleurs réalisateurs par score moyen pondéré
    df_directors = df_analysis[df_analysis['primaryProfession'].str.contains('director', na=False)].copy()
    df_directors['averageRating'] = pd.to_numeric(df_directors['averageRating'], errors='coerce')
    df_directors['numVotes'] = pd.to_numeric(df_directors['numVotes'], errors='coerce')
    df_directors = df_directors.dropna(subset=['averageRating', 'numVotes'])
    C = df_directors['averageRating'].mean()
    m = df_directors['numVotes'].quantile(0.75)
    df_directors['weighted_rating'] = (
        df_directors['numVotes'] / (df_directors['numVotes'] + m) * df_directors['averageRating'] +
        m / (df_directors['numVotes'] + m) * C
    )
    director_scores = df_directors.groupby('primaryName').agg({'weighted_rating': 'mean', 'numVotes': 'sum'}).reset_index()
    top_10_weighted_directors = director_scores.nlargest(10, 'weighted_rating')
    fig3, ax3 = plt.subplots(figsize=(12, 8))
    ax3.barh(top_10_weighted_directors['primaryName'], top_10_weighted_directors['weighted_rating'], color='skyblue')
    ax3.set_xlabel('Score moyen pondéré', fontsize=18, fontweight='bold', family='serif')
    ax3.set_ylabel('Réalisateur', fontsize=14, fontweight='bold')
    ax3.set_title('Les 10 meilleurs réalisateurs par score moyen pondéré', fontsize=14, fontweight='bold')
    ax3.invert_yaxis()

    # 4. Top 10 des acteurs les plus actifs
    df_actors = df_analysis[df_analysis['primaryProfession'].str.contains('actor|actress', na=False)]
    acteur_count = df_actors.groupby(['nconst', 'primaryName']).size().reset_index(name='nombre_films')
    acteur_count = acteur_count.sort_values(by='nombre_films', ascending=False)
    top_10_acteurs = acteur_count.head(10)
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    ax4.barh(top_10_acteurs['primaryName'], top_10_acteurs['nombre_films'], color='brown')
    ax4.set_xlabel('Nombre de films', fontsize=18, fontweight='bold', family='serif')
    ax4.set_ylabel('Acteurs', fontsize=14, fontweight='bold')
    ax4.set_title('Top 10 des acteurs les plus actifs', fontsize=14, fontweight='bold')
    ax4.invert_yaxis()

    # 5. Nombre de films par les 10 réalisateurs les plus prolifiques
    df_directors = df_analysis[df_analysis['primaryProfession'].str.contains('director', na=False)]
    director_count = df_directors['primaryName'].value_counts().reset_index()
    director_count.columns = ['primaryName', 'nombre_films']
    top_10_directors = director_count.head(10)
    fig5, ax5 = plt.subplots(figsize=(12, 8))
    ax5.barh(top_10_directors['primaryName'], top_10_directors['nombre_films'], color='lightblue')
    ax5.set_xlabel('Nombre de films', fontsize=18, fontweight='bold', family='serif')
    ax5.set_ylabel('Réalisateur', fontsize=14, fontweight='bold')
    ax5.set_title('Nombre de films par les 10 réalisateurs les plus prolifiques', fontsize=14, fontweight='bold')
    ax5.invert_yaxis()

    # 7. Distribution de genre des acteurs et actrices
    actor_count = df_analysis['primaryProfession'].str.contains('actor', na=False) & ~df_actors['primaryProfession'].str.contains('actress', na=False)
    actress_count = df_analysis['primaryProfession'].str.contains('actress', na=False)
    num_actors = actor_count.sum()
    num_actresses = actress_count.sum()

    labels = ['Acteurs (Hommes)', 'Actrices (Femmes)']
    sizes = [num_actors, num_actresses]
    colors = ['blue', 'pink']
    explode = (0.1, 0)

    fig6, ax6 = plt.subplots(figsize=(8, 8))

    # Fonction pour formater le texte des pourcentages
    def format_autopct(pct):
        return f'{pct:.1f}%'

    # Ajuste les tailles des labels et des pourcentages
    ax6.pie(sizes, explode=explode, labels=labels, colors=colors, autopct=format_autopct, startangle=140,
            textprops={'fontsize': 10, 'fontweight': 'bold'})

    # Titre et configuration des labels
    ax6.set_title('Distribution de Genre des Acteurs et Actrices', fontsize=12, fontweight='bold')
    ax6.axis('equal')  # Assure que le camembert est dessiné en cercle

   # Filtrer les lignes par genre (en supposant qu'il existe une colonne primaryProfession)
    genres = ['actor', 'actress']
    df_actors = df_analysis[df_analysis['primaryProfession'].str.contains('|'.join(genres), na=False)].copy()

    # Compter le nombre d'acteurs et d'actrices
    actor_count = df_actors['primaryProfession'].str.contains('actor', na=False) & ~df_actors['primaryProfession'].str.contains('actress', na=False)
    actress_count = df_actors['primaryProfession'].str.contains('actress', na=False)

    num_actors = actor_count.sum()
    num_actresses = actress_count.sum()

    # Créer un diagramme à barres horizontales pour visualiser la distribution de genre
    labels = ['Acteurs (Hommes)', 'Actrices (Femmes)']
    sizes = [num_actors, num_actresses]
    colors = ['blue', 'pink']

    # Crée un graphique en barres horizontales avec Matplotlib
    fig7, ax7 = plt.subplots(figsize=(10, 6))
    ax7.barh(labels, sizes, color=colors)
    ax7.set_xlabel('Nombre', fontsize=18, fontweight='bold', family='serif')
    ax7.set_title('Distribution de Genre des Acteurs et Actrices', fontsize=14, fontweight='bold')
    for index, value in enumerate(sizes):
        ax7.text(value, index, str(value), va='center')  # Afficher le nombre de chaque barre

    # Filtrer les lignes pour les scénaristes
    df_writers = df_analysis[df_analysis['primaryProfession'].str.contains('writer', na=False)].copy()

    # Compter le nombre de films par scénariste
    films_per_writer = df_writers['primaryName'].value_counts().reset_index()
    films_per_writer.columns = ['primaryName', 'numFilms']

    # Afficher les 10 scénaristes les plus prolifiques
    top_10_writers = films_per_writer.head(10)

    # Créer un diagramme à barres pour visualiser le nombre de films par scénariste
    fig8, ax8 = plt.subplots(figsize=(12, 8))
    ax8.barh(top_10_writers['primaryName'], top_10_writers['numFilms'], color='blue')
    ax8.set_xlabel('Nombre de films', fontsize=18, fontweight='bold', family='serif')
    ax8.set_ylabel('Scénariste', fontsize=14, fontweight='bold')
    ax8.set_title('Top 10 des Scénaristes par Nombre de Films', fontsize=14, fontweight='bold')
    ax8.invert_yaxis()  # Inverser l'axe y pour avoir le scénariste avec le plus de films en haut

    # Calculer le pourcentage de films adultes
    total_films = len(df_analysis)
    adult_films_count = df_analysis['isAdult'].sum()
    non_adult_films_count = total_films - adult_films_count
    adult_films_percentage = (adult_films_count / total_films) * 100

    # Créer un DataFrame pour le barplot
    barplot_data = pd.DataFrame({
        'Category': ['Adult', 'Non-Adult'],
        'Count': [adult_films_count, non_adult_films_count],
        'Percentage': [adult_films_percentage, 100 - adult_films_percentage]
    })

    # Créer un graphique à barres pour le pourcentage de films adultes
    fig9, ax9 = plt.subplots(figsize=(10, 6))
    barplot = sns.barplot(data=barplot_data, x='Category', y='Count', palette=['orange', 'blue'], ax=ax9)

    # Ajouter des annotations sur les barres
    for index, row in barplot_data.iterrows():
        barplot.text(row.name, row['Count'], f"{row['Percentage']:.1f}%", color='black', ha="center")

    ax9.set_title('Pourcentage de Films Adultes', fontsize=14, fontweight='bold')
    ax9.set_xlabel('Catégorie', fontsize=18, fontweight='bold', family='serif')
    ax9.set_ylabel('Nombre de films', fontsize=14, fontweight='bold')

     # Extraire les top 10 films les mieux notés
    top_rated_movies = df_analysis.nlargest(10, 'averageRating')[['title', 'averageRating', 'numVotes']]


    # Centrer le titre avec Markdown
    st.markdown("<h5 style='text-align: center;'>Top 10 des films les mieux notés</h1>", unsafe_allow_html=True)

    # Centrer le tableau avec une technique CSS
    st.markdown("""
    <style>
    .dataframe {
        margin-left: auto;
        margin-right: auto;
        border-collapse: collapse;
        width: 80%;  /* Ajuste la largeur du tableau si nécessaire */
    }
    .dataframe th, .dataframe td {
        padding: 8px;
        text-align: center;
    }
    .dataframe th {
        background-color: #f2f2f2;
    }
    .dataframe tr:nth-child(even) {
        background-color: #f9f9f9;
    }
    </style>
    """, unsafe_allow_html=True)

    # Afficher le tableau avec HTML et CSS
    st.write(top_rated_movies.to_html(classes='dataframe', index=False), unsafe_allow_html=True)

    # Plot du top 10 des films les mieux notés
    fig10, ax10 = plt.subplots(figsize=(10, 6))
    sns.barplot(x='averageRating', y='title', data=top_rated_movies, palette='viridis', ax=ax10)
    ax10.set_title('Top 10 des Films les Mieux Notés', fontsize=14, fontweight='bold')
    ax10.set_xlabel('Note Moyenne', fontsize=18, fontweight='bold', family='serif')
    ax10.set_ylabel('Titre du Film', fontsize=14, fontweight='bold')
    plt.tight_layout()  # Pour s'assurer que rien n'est coupé dans le graphique

    # Remplacer les valeurs infinies par NaN dans toutes les colonnes
    df_analysis = df_analysis.replace([float('inf'), float('-inf')], pd.NA)

    # Filtrer les données pour ne garder que les films
    # films_df = df_analysis[df_analysis['titleType'] == 'movie']

    # Supprimer les lignes avec des valeurs manquantes dans 'startYear' et 'runtimeMinutes'
    films_df = df_analysis.dropna(subset=['startYear', 'runtimeMinutes'])

    # Grouper les données par année de sortie et calculer la durée moyenne
    average_runtime_per_year = films_df.groupby('startYear')['runtimeMinutes'].mean()

    # Créer un graphique des durées moyennes par année de sortie
    fig11, ax11 = plt.subplots(figsize=(12, 6))
    sns.lineplot(data=average_runtime_per_year, marker='o', ax=ax11)
    ax11.set_title('Durée Moyenne des Films par Année de Sortie', fontsize=14, fontweight='bold')
    ax11.set_xlabel('Année de Sortie', fontsize=18, fontweight='bold', family='serif')
    ax11.set_ylabel('Durée Moyenne en Minutes', fontsize=14, fontweight='bold')
    ax11.grid(False)  # Ajouter une grille pour une meilleure lisibilité


    # Définir la période de filtrage
    start_period = 1990
    end_period = 2000

    # Filtrer les données pour ne garder que les films et une période donnée
    films_df = df_analysis[(df_analysis['titleType'] == 'movie') &
                  (df_analysis['startYear'] >= start_period) &
                  (df_analysis['startYear'] <= end_period)]

    # Vérifier si le DataFrame filtré n'est pas vide
    if films_df.empty:
        st.warning(f"Aucun film trouvé entre {start_period} et {end_period}.")
    else:
        # Extraction et comptage des genres
        genres_series = films_df['genres'].dropna().str.split(',').explode()

        # Vérifier si la série des genres n'est pas vide
        if genres_series.empty:
            st.warning("Aucun genre trouvé pour les films sélectionnés.")
        else:
            genre_counts = genres_series.value_counts().reset_index()
            genre_counts.columns = ['genre', 'count']

            # Créer un histogramme pour la répartition des genres
            fig12, ax12 = plt.subplots(figsize=(10, 6))
            sns.barplot(data=genre_counts, x='genre', y='count', ax=ax12)
            ax12.set_title(f"Répartition des Genres de Films de {start_period} à {end_period}")
            ax12.set_xlabel('Genres')
            ax12.set_ylabel('Nombre de Films')
            ax12.set_xticklabels(ax12.get_xticklabels(), rotation=45)
            ax12.grid(axis='y')


    

    return [fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8, fig9, fig10, fig11, fig12]

def main():


    st.title("Système de recommandation de films")

    # Chargement des données et préparation du modèle
    df_recommendation = load_recommendation_data()
    df_analysis = load_analysis_data()

    # Préparation du modèle de recommandation
    numeric_columns = df_recommendation.select_dtypes(include='number').columns
    X = df_recommendation[numeric_columns]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    knn_model = NearestNeighbors(n_neighbors=6, metric='cosine')
    knn_model.fit(X_scaled)

    # Sélecteur de film avec clé unique
    movie_title = st.selectbox("Choisissez un film", df_recommendation['title'], key="movie_selectbox")

    if st.button("Recommander"):
        recommendations = recommend_movies(movie_title, df_recommendation, knn_model, X_scaled)
        if not recommendations.empty:
            st.write(f"Recommandations pour le film '{movie_title}':")
            for _, row in recommendations.iterrows():
                st.subheader(row['title'])
                movie_id = row['tconst']
                poster_url = get_poster_path(df_recommendation, movie_id)
                start_year = row.get('startYear', 'N/A')
                try:
                    start_year = int(start_year)
                except (ValueError, TypeError):
                    start_year = 'N/A'
                st.markdown(f"""
                    <div style="display: flex; align-items: center;">
                        <img src="{poster_url}" width="300" style="margin-right: 20px;" />
                        <div>
                            <div><strong>Année de sortie:</strong> {start_year}</div>
                            <div><strong>Note moyenne:</strong> {row.get('averageRating', 'N/A')}</div>
                            <div><strong>Durée:</strong> {row.get('runtimeMinutes', 'N/A')} minutes</div>
                            <div><strong>Nombre de votes:</strong> {row.get('numVotes', 'N/A')}</div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.write("Aucune recommandation trouvée pour ce film.")

    # Création des graphiques
    st.title("Analyse sur les films recommandés")
    figures = create_graphs(df_analysis)

    # Sélectionner aléatoirement trois graphiques
    selected_figures = random.sample(figures, 3)

    # Afficher les graphiques sélectionnés
    for fig in selected_figures:
        st.pyplot(fig)

if __name__ == "__main__":
    main()
