import pandas as pd
import argparse
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

import mlflow
import mlflow.sklearn

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="rf")
parser.add_argument("--n_estimators", type=int, default=100)
parser.add_argument("--max_depth", type=int, default=5)
parser.add_argument("--feature_subset", type=int, default=0)
parser.add_argument("--data", type=str, default="data/winequality-red.csv")

args = parser.parse_args()

# Load dataset
df = pd.read_csv(args.data, sep=';')

# Clean columns
df.columns = df.columns.str.strip().str.replace(" ", "_")

# Convert target to binary
df['quality'] = df['quality'].apply(lambda x: 1 if x >= 7 else 0)

# Feature selection
if args.feature_subset == 1:
    df = df[['alcohol', 'pH', 'sulphates', 'quality']]

X = df.drop("quality", axis=1)
y = df["quality"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model selection
if args.model == "rf":
    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth
    )
else:
    model = LogisticRegression(max_iter=1000)

# MLflow setup
mlflow.set_experiment("2022BCS0098_experiment")

with mlflow.start_run():
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/model.pkl")

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    # Log params
    mlflow.log_param("model", args.model)
    mlflow.log_param("n_estimators", args.n_estimators)
    mlflow.log_param("max_depth", args.max_depth)
    mlflow.log_param("feature_subset", args.feature_subset)

    # Log metrics
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)

    # Log model
    mlflow.sklearn.log_model(model, "model")

    print("Accuracy:", acc)
    print("F1 Score:", f1)
