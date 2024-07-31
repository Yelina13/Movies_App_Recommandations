import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

# Fonction pour charger les données de recommandation
@st.cache
def load_recommendation_data():
    """Charger le jeu de données pour les recommandations de films."""
    try:
        df = pd.read_csv("data1.csv")  # Remplacez par le chemin vers votre fichier
        # Nettoyer les valeurs NaN dans la colonne 'poster_path'
        df = df.dropna(subset=['poster_path'])
        return df
    except FileNotFoundError:
        st.error("Erreur : Le fichier data1.csv est introuvable. Vérifiez le chemin ou le nom du fichier.")
        return pd.DataFrame()

# Fonction pour charger les données d'analyse des acteurs
@st.cache
def load_analysis_data():
    """Charger le jeu de données pour l'analyse des acteurs."""
    try:
        return pd.read_csv('movies_france_2000.csv', low_memory=False)
    except FileNotFoundError:
        st.error("Erreur : Le fichier movies_france_2000.csv est introuvable. Vérifiez le chemin ou le nom du fichier.")
        return pd.DataFrame()

# Fonction pour obtenir le chemin de l'affiche
def get_poster_path(df, movie_id):
    poster_path = df[df['tconst'] == movie_id]['poster_path'].values[0]
    full_path = "https://image.tmdb.org/t/p/w500/" + poster_path
    return full_path

# Fonction de recommandation de films
def recommend_movies(movie_title, df, knn_model, X_scaled):
    """Recommander des films basés sur le titre d'un film donné."""
    try:
        movie_index = df[df["title"] == movie_title].index[0]
        _, indices = knn_model.kneighbors([X_scaled[movie_index]])
        recommended_movies_index = indices[0][1:]
        recommendations = df.iloc[recommended_movies_index][['title', 'poster_path']]
        
        # Formatage des URLs des affiches en utilisant la fonction get_poster_path
        recommendations['poster_urls'] = recommendations['title'].apply(lambda title: get_poster_path(title, df))
        
        return recommendations
    except (IndexError, KeyError):
        st.error("Erreur : Le titre du film est introuvable.")
        return pd.DataFrame(columns=['title', 'poster_urls'])

# Fonction principale de l'application Streamlit
def main():
    """Fonction principale pour l'application Streamlit."""
    st.title("Système de recommandation de films")

    # Charger les données de recommandation
    df = load_recommendation_data()
    if df.empty:
        st.error("Erreur : Les données de recommandation n'ont pas pu être chargées.")
        return

    if 'title' not in df.columns or len(df.columns) < 3:
        st.error("Erreur : Le format des données de recommandation est incorrect.")
        return

    # Préparer les données pour KNN
    columns_reco = df.columns[2:]
    X = df[columns_reco]

    # Nettoyage des données
    X = X.apply(pd.to_numeric, errors='coerce')
    X.replace([float('inf'), -float('inf')], pd.NA, inplace=True)

    # Débogage : Afficher les colonnes avec des NaN restants
    nan_columns = X.columns[X.isna().any()].tolist()
    if nan_columns:
        st.write("Colonnes contenant encore des valeurs NaN après nettoyage :")
        st.write(X[nan_columns].isna().sum())
    
    # Méthodes de remplacement des NaN
    X.fillna(X.median(), inplace=True)  # Utilisation de la médiane au lieu de la moyenne pour moins de biais

    # Vérifiez de nouveau s'il y a des NaN
    if X.isna().sum().sum() > 0:
        st.error(f"Erreur : Les données contiennent encore des valeurs NaN après nettoyage. Valeurs NaN restantes : {X.isna().sum().sum()}")
        return

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Créer et entraîner le modèle k-NN
    knn_model = NearestNeighbors(n_neighbors=6, metric='cosine')
    knn_model.fit(X_scaled)

    # Sélecteur de film
    movie_title = st.selectbox("Choisissez un film", df['title'])

    if st.button("Recommander"):
        with st.spinner("Chargement des recommandations..."):
            recommendations = recommend_movies(movie_title, df, knn_model, X_scaled)
        if not recommendations.empty:
            st.write(f"Recommandations pour le film '{movie_title}' :")
            for _, row in recommendations.iterrows():
                st.write(row['title'])
                st.image(row['poster_urls'], width=150)
        else:
            st.write("Aucune recommandation trouvée pour ce film. Essayez un autre titre.")

    # Section d'analyse des acteurs
    st.title("Analyse des Acteurs")

    # Charger les données d'analyse
    df_analysis = load_analysis_data()
    if df_analysis.empty:
        st.error("Erreur : Les données d'analyse des acteurs n'ont pas pu être chargées.")
        return

    # Calculer l'âge des acteurs
    df_analysis['birthYear'] = pd.to_numeric(df_analysis['birthYear'], errors='coerce')
    df_analysis = df_analysis.dropna(subset=['birthYear'])
    df_analysis['age'] = 2024 - df_analysis['birthYear']

    acteur_count = df_analysis['primaryName'].value_counts().reset_index()
    acteur_count.columns = ['primaryName', 'nombre_apparitions']
    top_10_acteurs = acteur_count.head(10)
    top_10_df = df_analysis[df_analysis['primaryName'].isin(top_10_acteurs['primaryName'])]
    age_moyen_top_10 = top_10_df.groupby('primaryName')['age'].mean().reset_index()

    fig1, ax1 = plt.subplots(figsize=(12, 8))
    sns.barplot(data=age_moyen_top_10, x='primaryName', y='age', order=top_10_acteurs['primaryName'], ax=ax1)
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
    ax1.set_xlabel('Acteur')
    ax1.set_ylabel('Âge moyen')
    ax1.set_title('Âge moyen des 10 acteurs les plus actifs')
    st.pyplot(fig1)

    # Répartition de l'âge par genre
    df_actors = df_analysis.copy()
    df_actors['deathYear'] = pd.to_numeric(df_actors['deathYear'], errors='coerce')
    df_actors = df_actors.dropna(subset=['birthYear'])
    df_actors['age'] = df_actors.apply(lambda row: (row['deathYear'] if pd.notna(row['deathYear']) else 2024) - row['birthYear'], axis=1)
    genres = ['actor', 'actress']
    df_actors = df_actors[df_actors['primaryProfession'].str.contains('|'.join(genres), na=False)]
    df_male_actors = df_actors[df_actors['primaryProfession'].str.contains('actor', na=False) & ~df_actors['primaryProfession'].str.contains('actress', na=False)]
    df_female_actors = df_actors[df_actors['primaryProfession'].str.contains('actress', na=False)]

    fig2, ax2 = plt.subplots(figsize=(12, 8))
    bins = range(0, 110, 5)
    ax2.hist(df_male_actors['age'], bins=bins, alpha=0.5, label='Acteurs (Hommes)', color='blue', edgecolor='black')
    ax2.hist(df_female_actors['age'], bins=bins, alpha=0.5, label='Actrices (Femmes)', color='red', edgecolor='black')
    ax2.set_xlabel('Âge')
    ax2.set_ylabel('Nombre d\'acteurs')
    ax2.set_title('Répartition de l\'âge des acteurs en fonction du genre')
    ax2.legend()
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    ax2.set_xticks(bins)
    st.pyplot(fig2)

    # Analyse des réalisateurs
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
    ax3.set_xlabel('Score moyen pondéré')
    ax3.set_ylabel('Réalisateur')
    ax3.set_title('Les 10 meilleurs réalisateurs par score moyen pondéré')
    ax3.invert_yaxis()
    st.pyplot(fig3)

    # Analyse des réalisateurs les plus prolifiques
    director_count = df_directors['primaryName'].value_counts().reset_index()
    director_count.columns = ['primaryName', 'nombre_films']
    top_10_directors = director_count.head(10)
    top_10_directors_df = df_directors[df_directors['primaryName'].isin(top_10_directors['primaryName'])]
    fig4, ax4 = plt.subplots(figsize=(12, 8))
    sns.barplot(data=top_10_directors_df, x='primaryName', y='numVotes', order=top_10_directors['primaryName'], ax=ax4)
    ax4.set_xticklabels(ax4.get_xticklabels(), rotation=45, ha='right')
    ax4.set_xlabel('Réalisateur')
    ax4.set_ylabel('Nombre de votes')
    ax4.set_title('Nombre de films des 10 réalisateurs les plus prolifiques')
    st.pyplot(fig4)

    # Distribution des genres des acteurs
    genres = ['actor', 'actress']
    df_actors = df_analysis[df_analysis['primaryProfession'].str.contains('|'.join(genres), na=False)]
    actor_count = df_actors['primaryProfession'].str.contains('actor', na=False) & ~df_actors['primaryProfession'].str.contains('actress', na=False)
    actress_count = df_actors['primaryProfession'].str.contains('actress', na=False)
    num_actors = actor_count.sum()
    num_actresses = actress_count.sum()
    labels = ['Acteurs (Hommes)', 'Actrices (Femmes)']
    sizes = [num_actors, num_actresses]
    colors = ['blue', 'pink']
    explode = (0.1, 0)
    fig7, ax7 = plt.subplots(figsize=(8, 8))
    ax7.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
    ax7.set_title('Distribution de Genre des Acteurs et Actrices')
    ax7.axis('equal')
    st.pyplot(fig7)

if __name__ == "__main__":
    main()
