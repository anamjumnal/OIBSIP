"""Train + compare classifiers, save the best pipeline to model/spam_model.joblib"""
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, f1_score
from utils import load_data, clean_text

df = load_data().drop_duplicates()
df["clean"] = df["text"].apply(clean_text)
y = (df["label"] == "spam").astype(int)
X_tr, X_te, y_tr, y_te = train_test_split(df["clean"], y, test_size=0.2, stratify=y, random_state=42)

models = {
    "Multinomial NB": MultinomialNB(alpha=0.1),
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Linear SVM": LinearSVC(class_weight="balanced"),
}
best, best_f1 = None, -1
for name, clf in models.items():
    pipe = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=50000, sublinear_tf=True), clf).fit(X_tr, y_tr)
    pred = pipe.predict(X_te)
    f1 = f1_score(y_te, pred)
    print(f"\n=== {name} (F1={f1:.4f}) ===\n", classification_report(y_te, pred, target_names=["ham", "spam"]))
    if f1 > best_f1:
        best, best_f1, best_name = pipe, f1, name
# Probability-capable model for the web app: use Logistic Regression if best is SVM
joblib.dump({"model": best, "name": best_name}, "model/spam_model.joblib")
print("Saved best model:", best_name)
