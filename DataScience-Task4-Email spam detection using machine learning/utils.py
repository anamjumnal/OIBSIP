"""Shared helpers: dataset loading + text preprocessing (used by notebook, trainer and web app)."""
import io, os, re, string, zipfile, urllib.request
import pandas as pd
import nltk
from nltk.stem import PorterStemmer

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "emails.csv")
URL = "https://raw.githubusercontent.com/MWiechmann/enron_spam_data/master/enron_spam_data.zip"  # Enron email spam corpus (Kaggle/GitHub)

def load_data():
    """Download the Enron email spam dataset once, cache as data/emails.csv, return DataFrame[label, text]
    where text = subject + body of each email. (Manual option: put any CSV with columns label,text in data/emails.csv)"""
    if not os.path.exists(DATA_PATH):
        print("Downloading Enron email dataset (~15 MB)...")
        raw = urllib.request.urlopen(URL, timeout=120).read()
        z = zipfile.ZipFile(io.BytesIO(raw))
        d = pd.read_csv(z.open(z.namelist()[0]))
        d["text"] = d["Subject"].fillna("") + " . " + d["Message"].fillna("")
        d = d.rename(columns={"Spam/Ham": "label"})[["label", "text"]]
        d = d[d["text"].str.strip().str.len() > 3]
        os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
        d.to_csv(DATA_PATH, index=False)
    return pd.read_csv(DATA_PATH)

# ---- preprocessing ----
try:
    from nltk.corpus import stopwords
    try:
        STOP = set(stopwords.words("english"))
    except LookupError:
        nltk.download("stopwords", quiet=True)
        STOP = set(stopwords.words("english"))
except Exception:
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    STOP = set(ENGLISH_STOP_WORDS)

_stem = PorterStemmer()
_punct = str.maketrans("", "", string.punctuation)

def clean_text(text: str) -> str:
    """lowercase -> strip HTML/addresses/URLs -> remove punctuation/digits -> drop stopwords -> stem."""
    text = str(text)[:3000].lower()                      # emails can be long; first 3000 chars is plenty
    text = re.sub(r"<[^>]+>", " ", text)                 # strip HTML tags
    text = re.sub(r"\S+@\S+", " emailaddr ", text)        # email addresses
    text = re.sub(r"http\S+|www\S+", " url ", text)      # links
    text = text.translate(_punct)
    text = re.sub(r"\d+", " ", text)
    return " ".join(_stem.stem(w) for w in text.split() if w not in STOP and len(w) > 1)
