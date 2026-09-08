# Sentiment Analysis Project

A complete, self-contained sentiment analysis pipeline: dataset generation,
text preprocessing, model training/evaluation, a CLI predictor, and a
Flask web app — all runnable offline with standard Python data-science
libraries (no API keys or internet access needed at runtime).

The model classifies text into **positive / neutral / negative**.

## Project structure

```
sentiment_project/
├── data/
│   └── reviews.csv              # labeled dataset (text, sentiment)
├── models/
│   ├── sentiment_model.joblib   # trained LogisticRegression model
│   └── tfidf_vectorizer.joblib  # fitted TF-IDF vectorizer
├── outputs/
│   ├── confusion_matrix.png     # evaluation plot
│   └── evaluation_report.txt    # accuracy + precision/recall/F1
├── src/
│   ├── generate_data.py         # builds the labeled dataset
│   ├── preprocess.py            # text cleaning utilities
│   ├── train.py                 # trains + evaluates + saves the model
│   └── predict.py               # CLI to classify new text
├── templates/
│   └── index.html               # web UI (served by app.py)
├── app.py                       # Flask web app + JSON API
├── requirements.txt
└── README.md
```

## How it works

1. **Data** (`src/generate_data.py`): builds a balanced, 3-class dataset
   (720 rows, 240 per class) from review-style sentence templates across
   20 subjects (product, movie, restaurant, app, etc). This keeps the
   whole project runnable with zero external downloads. Swap in your own
   `data/reviews.csv` (two columns: `text`, `sentiment`) to use real data —
   everything downstream works unchanged.
2. **Preprocessing** (`src/preprocess.py`): lowercases text, strips
   punctuation/numbers, and removes a small stopword list. Negation words
   like "not"/"no" are deliberately *kept* since they flip sentiment.
3. **Model** (`src/train.py`): TF-IDF vectorization (unigrams + bigrams)
   feeding a Logistic Regression classifier — a fast, strong, and easy to
   explain baseline for text classification. Trains on an 80/20 split,
   prints/saves a classification report, and saves a confusion matrix plot.
4. **Inference**: `src/predict.py` (CLI) and `app.py` (web app + JSON API)
   both load the saved model/vectorizer and classify new text.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### 1. Regenerate the dataset (optional — already included)
```bash
python src/generate_data.py
```

### 2. Train the model
```bash
python src/train.py
```
This prints accuracy and a classification report, and writes
`outputs/confusion_matrix.png` + `outputs/evaluation_report.txt`.

### 3. Predict from the command line
```bash
python src/predict.py "The food was cold and the service was slow"
# Predicted sentiment: negative

python src/predict.py     # interactive mode, type sentences one at a time
```

### 4. Run the Streamlit Web App (Recommended)
```bash
streamlit run streamlit_app.py
```
Open the local URL (usually **http://localhost:8501**) in your browser.
The Streamlit app includes:
- **Live Predictor**: Type any sentence or pick quick test presets to view real-time confidence breakdowns and token extraction details.
- **Batch CSV Analysis**: Upload any CSV, choose the text column, run bulk classification, inspect class distributions, and download the annotated CSV.
- **Model Performance & Dataset Explorer**: View model evaluation report, confusion matrix plot, and browse training data.

### 5. Run the Flask Web App (Alternative)
```bash
python app.py
```
Open **http://127.0.0.1:5000** — lightweight Flask interface + JSON REST API.

JSON API:
```bash
curl -X POST http://127.0.0.1:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I really enjoyed this, great job!"}'
```

### 6. Deploy to Streamlit Community Cloud (Free Hosting)
You can deploy this application directly to **Streamlit Community Cloud** in 3 steps:

1. **Push your code to GitHub**:
   ```bash
   git init
   git add .
   git commit -m "Deploy Sentiment Analysis on Streamlit"
   git branch -M main
   git remote add origin https://github.com/<YOUR_GITHUB_USERNAME>/<YOUR_REPOSITORY_NAME>.git
   git push -u origin main
   ```
2. **Go to [share.streamlit.io](https://share.streamlit.io)** and log in with your GitHub account.
3. Click **New app**, select your repository, set the branch to `main`, and set the **Main file path** to `streamlit_app.py`.
4. Click **Deploy!** Streamlit Cloud will automatically install dependencies from `requirements.txt` and launch your live application with a public URL.

## Results

On the held-out 20% test split, the model reaches **100% accuracy** —
expected, since the dataset is built from a fixed set of sentence
templates that are highly separable by vocabulary. On genuinely novel,
hand-written sentences (not from any template) it performs well but not
perfectly, e.g. correctly flags negation and mixed-sentiment phrasing,
though very subtle/ambiguous cases (e.g. faint praise) can be a close call.

## Extending this project

- **Use real data**: swap `data/reviews.csv` for a real labeled dataset
  (e.g. product reviews, tweets) — no other code changes needed as long
  as it has `text` and `sentiment` columns.
- **Try a different model**: swap `LogisticRegression` in `train.py` for
  `LinearSVC`, `MultinomialNB`, or a fine-tuned transformer
  (e.g. via Hugging Face) for higher accuracy on harder, real-world text.
- **Add more classes**: e.g. a 5-point scale (very negative → very
  positive) by relabeling the dataset.
- **Deploy**: the Flask app is ready to containerize (Docker) or deploy
  behind gunicorn for a small production service.
