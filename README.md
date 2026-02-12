# Mindminer

Content-based **movie recommendation** system: get similar movies based on tags (genres, keywords, cast, crew). Includes preprocessing, training, EDA, and a Tkinter UI.

## Features

- **Preprocessing:** Merge TMDB movies + credits, clean and build a unified `tags` column.
- **Recommendation:** Cosine similarity on tag vectors (CountVectorizer + NLTK stemming); **case-insensitive** movie search.
- **EDA:** Exploratory notebooks (title length, tag distribution, top tags, etc.).
- **UI:** Simple desktop app (Tkinter) to search by movie name and see top-5 similar movies.

## Sample output

**UI (Mindminer app)**  
Enter a movie title (e.g. `Avatar` or `avatar`) and click **Search** (or press Enter). The app shows the 5 most similar movies by tag similarity.

Example: input **Avatar** → recommendations might include:

```
Titan A.E.
Aliens
Star Wars: The Clone Wars
Battle for Terra
...
```

*(Exact titles depend on your dataset.)*

Optional: add a screenshot of the UI and reference it here, e.g.:

<!--
![Mindminer UI](docs/screenshot.png)
Place a file at docs/screenshot.png (or ./screenshot.png) and uncomment the line above.
-->

## Project structure

```
├── data/
│   ├── raw/              # movies.csv, credits.csv
│   └── processed/         # movies_merge.csv (output of preprocessing)
├── notebooks/
│   └── EDA.ipynb          # Exploratory data analysis
├── src/
│   ├── preprocessing/
│   │   └── preprocessing.ipynb
│   ├── training/
│   │   ├── training.ipynb
│   │   └── recommender.py # recommend(movie) used by UI
│   └── ui/
│       └── ui.ipynb       # Tkinter GUI
├── requirements.txt
└── README.md
```

## Setup

1. **Clone** and go to the project root.
2. **Virtual environment** (recommended):

   ```bash
   python -m venv venv
   venv\Scripts\activate   # Windows
   # source venv/bin/activate   # Linux / macOS
   ```
3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```
4. **NLTK data** (if you get tokenizer errors):

   ```python
   import nltk
   nltk.download("punkt")
   ```

## Data

- Put **raw** TMDB files in `data/raw/`:
  - `movies.csv`
  - `credits.csv`
- Run **preprocessing** to generate `data/processed/movies_merge.csv`.
- The **recommender** and **UI** read from `data/processed/movies_merge.csv`.

## How to run

1. **Preprocessing**Open and run `src/preprocessing/preprocessing.ipynb` to produce `movies_merge.csv` in `data/processed/`.
2. **Training** (optional)`src/training/training.ipynb` builds the similarity logic. The UI uses `src/training/recommender.py`, which loads the same data and builds the model on first use.
3. **EDA**Run `notebooks/EDA.ipynb` for visualizations (uses `data/processed/movies_merge.csv`).
4. **UI**
   Run `src/ui/ui.ipynb`: execute the **import** cell first, then the **Tkinter** cell.
   Run the notebook from the **project root** (or ensure the project root is in `sys.path`) so `from src.training.recommender import recommend` works.
   Search is **case-insensitive** (e.g. "avatar" or "Avatar" both work).
