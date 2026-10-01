# Spam Detector (NLP) - Notebook + Web App
```bash
python -m venv venv && source venv/bin/activate     # Windows: venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook spam_detection.ipynb               # run all cells (downloads the Enron email dataset, trains, saves model)
python app.py                                       # open http://127.0.0.1:5000
```
- `utils.py` data loader + preprocessing  - `train_model.py` CLI trainer (also auto-runs if no model exists)
- `app.py` Flask API (`POST /predict {"text": "..."}`)  - `templates/index.html` UI

**Dataset:** Enron email spam corpus (~33k real emails, subject+body), auto-downloaded from GitHub (also on Kaggle: "Enron Spam"). To use your own, put a CSV with columns `label` (spam/ham) and `text` at `data/emails.csv`.
