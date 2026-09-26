from flask import Flask, render_template, request, jsonify
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
import pandas as pd
import numpy as np

app = Flask(__name__)

# ============================================================
# LOAD IRIS DATASET
# ============================================================

iris = load_iris()

X = iris.data
y = iris.target

feature_names = [
    "Sepal Length",
    "Sepal Width",
    "Petal Length",
    "Petal Width"
]

target_names = [
    "Setosa",
    "Versicolor",
    "Virginica"
]

# Create DataFrame
df = pd.DataFrame(X, columns=feature_names)
df["Species"] = [target_names[i] for i in y]


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ============================================================
# MODELS
# ============================================================

models = {

    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )
}


# ============================================================
# TRAIN MODELS + EVALUATION
# ============================================================

results = {}

for model_name, model in models.items():

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    matrix = confusion_matrix(
        y_test,
        predictions
    )

    results[model_name] = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": matrix.tolist()
    }


# ============================================================
# BEST MODEL
# ============================================================

best_model_name = max(
    results,
    key=lambda name: results[name]["f1"]
)

best_model = models[best_model_name]


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def home():
    return render_template("index.html")


# ============================================================
# DATA + MODEL INFORMATION API
# ============================================================

@app.route("/api/data")
def get_data():

    # Average measurements for each species
    averages = {}

    for i, species in enumerate(target_names):

        species_data = X[y == i]

        averages[species] = [
            round(float(value), 3)
            for value in species_data.mean(axis=0)
        ]

    # Random Forest feature importance
    rf_model = models["Random Forest"]

    feature_importance = [
        round(float(value), 4)
        for value in rf_model.feature_importances_
    ]

    # Dataset preview
    preview = []

    for _, row in df.head(10).iterrows():

        preview.append({
            "sepal_length": row["Sepal Length"],
            "sepal_width": row["Sepal Width"],
            "petal_length": row["Petal Length"],
            "petal_width": row["Petal Width"],
            "species": row["Species"]
        })

    return jsonify({

        "samples": len(df),

        "features": len(feature_names),

        "classes": len(target_names),

        "feature_names": feature_names,

        "target_names": target_names,

        "averages": averages,

        "feature_importance": feature_importance,

        "preview": preview,

        "results": results,

        "best_model": best_model_name
    })


# ============================================================
# PREDICTION API
# ============================================================

@app.route("/api/predict", methods=["POST"])
def predict():

    try:

        data = request.get_json()

        sepal_length = float(data["sepal_length"])
        sepal_width = float(data["sepal_width"])
        petal_length = float(data["petal_length"])
        petal_width = float(data["petal_width"])

        input_data = np.array([
            [
                sepal_length,
                sepal_width,
                petal_length,
                petal_width
            ]
        ])

        prediction = best_model.predict(input_data)[0]

        probabilities = best_model.predict_proba(
            input_data
        )[0]

        return jsonify({

            "success": True,

            "species": target_names[prediction],

            "model": best_model_name,

            "probabilities": [
                round(float(p), 4)
                for p in probabilities
            ]
        })

    except Exception as error:

        return jsonify({
            "success": False,
            "error": str(error)
        }), 400


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )