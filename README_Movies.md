# Movies App – Filmempfehlungen

Eine Python-Anwendung, die auf Basis von Filmdaten **Filmempfehlungen** erstellt.

> **Kurzbeschreibung:** [In 1–2 Sätzen beschreiben: Wie funktioniert die Empfehlung (z. B. ähnliche Filme nach Genre, Schauspielern, Regie oder Bewertung)? Für wen ist die App gedacht?]

## Inhaltsverzeichnis

- [Funktionen](#funktionen)
- [Technologien](#technologien)
- [Projektstruktur](#projektstruktur)
- [Voraussetzungen](#voraussetzungen)
- [Installation](#installation)
- [Anwendung starten](#anwendung-starten)
- [Daten](#daten)
- [Autorin](#autorin)

## Funktionen

- [Funktion 1, z. B. Suche nach einem Film]
- [Funktion 2, z. B. Anzeige ähnlicher Filme als Empfehlung]
- [Funktion 3, z. B. Filter nach Genre oder Jahr]

## Technologien

- **Sprache:** Python
- **Weboberfläche:** [Streamlit – bitte prüfen und bei Bedarf anpassen]
- **Datenverarbeitung:** Pandas
- **Empfehlungslogik:** [z. B. scikit-learn, Nearest Neighbors – bitte anpassen]

## Projektstruktur

```
movies_app.py            Hauptdatei der Anwendung
requirements.txt         Liste der benötigten Python-Pakete
data.csv                 Filmdaten
data_others.csv          Zusätzliche Daten
movies_france_2000.csv   Datensatz zu französischen Filmen ab 2000
README.md                Projektbeschreibung
```

## Voraussetzungen

- [Python 3](https://www.python.org/downloads/)
- pip (wird mit Python installiert)

## Installation

1. Repository klonen:

   ```bash
   git clone https://github.com/Yelina13/Movies_App_Recommandations.git
   cd Movies_App_Recommandations
   ```

2. Virtuelle Umgebung erstellen und aktivieren:

   ```bash
   python -m venv venv
   source venv/bin/activate      # Linux/Mac
   venv\Scripts\activate         # Windows
   ```

3. Abhängigkeiten installieren:

   ```bash
   pip install -r requirements.txt
   ```

## Anwendung starten

```bash
streamlit run movies_app.py
```

Die Anwendung öffnet sich anschließend im Browser, in der Regel unter `http://localhost:8501`.

## Daten

Die Anwendung nutzt mehrere CSV-Dateien mit Filminformationen:

- `data.csv`: [Inhalt beschreiben, z. B. Titel, Genre, Bewertung]
- `data_others.csv`: [Inhalt beschreiben]
- `movies_france_2000.csv`: [Inhalt beschreiben]

Quelle der Daten: [Quelle angeben, z. B. IMDb oder Kaggle]

## Autorin

**Yelina13**
GitHub: [github.com/Yelina13](https://github.com/Yelina13)
