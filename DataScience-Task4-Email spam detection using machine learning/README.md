# 📧 Task 4: Email Spam Detection with Machine Learning

An NLP binary classifier that tells **spam** emails from legitimate (**ham**) emails, with a Jupyter Notebook for the full analysis and a pink-themed Flask web app for live predictions.

## Features
- Class distribution check (counts and percentages)
- Text preprocessing: lowercase, HTML, email address and URL removal, punctuation removal, stopword removal (NLTK) and stemming
- TF-IDF feature extraction (unigrams and bigrams)
- Three classifiers: Multinomial Naive Bayes, Logistic Regression and Linear SVM
- Evaluation with accuracy, precision, recall, F1-score and confusion matrices
- Discussion of why recall matters for spam detection
- WordCloud visualisations of spam and ham words
- Web app that classifies any pasted email with a confidence score

## Tech Stack
Python · pandas · scikit-learn · NLTK · matplotlib · seaborn · WordCloud · Flask · Jupyter Notebook

## Dataset
**Enron Email Spam corpus**: about 33,000 real emails (subject and body), roughly half spam and half ham. After removing duplicates, 30,493 emails remain. It downloads automatically the first time you run the project and is saved as `data/emails.csv`.

To use your own data, put a CSV with the columns `label` (spam or ham) and `text` at `data/emails.csv`.

## Results
Test set: 6,099 emails (20% stratified split).

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Multinomial Naive Bayes | 0.9885 | 0.9860 | 0.9901 | 0.9880 |
| Logistic Regression | 0.9872 | 0.9788 | 0.9949 | 0.9867 |
| **Linear SVM (best)** | **0.9921** | **0.9874** | **0.9962** | **0.9918** |

## Project Structure
```
├── app.py                 # Flask web app
├── utils.py               # dataset loader and text preprocessing
├── train_model.py         # trains the models and saves the best one
├── spam_detection.ipynb   # full notebook, step by step
├── requirements.txt
├── templates/
│   └── index.html         # web app page
├── data/                  # created automatically (dataset)
└── model/                 # created automatically (trained model)
```

## How to Run

**1. Create and activate a virtual environment**
```bash
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # Mac / Linux
```

**2. Install the libraries**
```bash
pip install -r requirements.txt
```

**3. Run the notebook**
```bash
jupyter notebook spam_detection.ipynb
```
Click **Run All**. The first run downloads the dataset and trains the models, which takes 2 to 4 minutes. It also saves the best model to `model/spam_model.joblib`.

**4. Start the web app**
```bash
python app.py
```
Open **http://127.0.0.1:5000**, paste an email and click **Check email**.
If no trained model exists yet, the app trains one automatically on the first start.

## Web App API
```
POST /predict
Content-Type: application/json

{"text": "Subject: You have won $1,000,000! Click here to claim..."}
```
Response:
```json
{"label": "spam", "confidence": 96.5}
```

## Limitations
- The legitimate emails come from a single company (Enron), so the model may do worse on emails written in a very different style.
- Very short or casual messages can be misclassified.

## Author
**Anam Jumnal**: Data Science Internship, Oasis Infobyte (OIBSIP)
