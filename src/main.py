import sys
sys.path.append("/Users/annadiaw/Desktop/NLP")

import click
import pandas as pd

from src.model.dumb_model import DumbModel
from src.data.make_dataset import make_dataset
from src.features.make_features import make_features
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split

@click.group()
def cli():
    pass

@click.command()
@click.option("--task", help="Can be is_comic_video, is_name, or find_comic_name")
@click.option("--input_file", default="src/data/raw/train.csv", help="Input file")
@click.option("--model_dump", default="src/model", help="Output directory for model dumps")
def train(task, input_file, model_dump):
    df = make_dataset(input_file)

    for model_name in ["LogisticRegression",  "RandomForest"]:
        for vectorizer_type in ["count", "hashing_vectorizer", "tfidf_vectorizer"]:
            print(f"Training model: {model_name}, vectorizer: {vectorizer_type}")
            model = DumbModel(model_name=model_name, vectorizer_type=vectorizer_type)
            model.df = df
            model_name_formatted = model_name.lower().replace(" ", "_")
            model.fit(df)
            model_dump_filename = f"{model_dump}/dump_{model_name_formatted}_{vectorizer_type}.json"
            model.dump(model_dump_filename)

@click.command()
@click.option("--task", help="Can be is_comic_video, is_name, or find_comic_name")
@click.option("--input_file", default="src/data/raw/train.csv", help="Input file")
@click.option("--model_dump", default="src/model", help="Directory containing model dumps")
@click.option("--output_dir", default="src/data/processed", help="Output directory for predictions")
def predict(task, input_file, model_dump, output_dir):
    df = pd.read_csv(input_file)
    for model_name in ["LogisticRegression", "RandomForest"]:
        for vectorizer_type in ["count", "hashing_vectorizer", "tfidf_vectorizer"]:
            print(f"Predicting model: {model_name}, vectorizer: {vectorizer_type}")
            model = DumbModel(model_name=model_name, vectorizer_type=vectorizer_type)
            model_name_formatted = model_name.lower().replace(" ", "_")
            model_dump_filename = f"{model_dump}/dump_{model_name_formatted}_{vectorizer_type}.json"
            model.load(model_dump_filename)
            
            # Même type de vectorizer que pour l'entraînement (count, hashing_vectorizer, tfidf_vectorizer)
            vectorizer_type = model.vectorizer_type
            X, _ = make_features(df, vectorizer_type=vectorizer_type)
            
            predictions = model.predict(df)
            df["prediction"] = predictions
            output_filename = f"{output_dir}/predictions_{model_name_formatted}_{vectorizer_type}.csv"
            df.to_csv(output_filename, index=False)

@click.command()
@click.option("--task", help="Can be is_comic_video, is_name, or find_comic_name")
@click.option("--input_file", default="src/data/raw/train.csv", help="Input file")
@click.option("--model_dump", default="src/model", help="Directory containing model dumps")
def evaluate(task, input_file, model_dump):
    if task == "is_comic_video":
        df = make_dataset(input_file)
        for model_name in ["LogisticRegression",  "RandomForest"]:
            for vectorizer_type in ["count", "hashing_vectorizer", "tfidf_vectorizer"]:
                print(f"Evaluating model: {model_name}, vectorizer: {vectorizer_type}")
                model = DumbModel(model_name=model_name, vectorizer_type=vectorizer_type)
                model_name_formatted = model_name.lower().replace(" ", "_")
                model_dump_filename = f"{model_dump}/dump_{model_name_formatted}_{vectorizer_type}.json"
                model.load(model_dump_filename)
                model.vectorizer_type = vectorizer_type
                model.df = df

                # Prépare les caractéristiques
                X, y = model.train(df)

                # Divise les données en ensembles d'entraînement et de test
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                model.X_train = X_train
                model.y_train = y_train

                # Entraîne votre modèle sur l'ensemble d'entraînement
                model.fit()

                # Prédise les étiquettes sur l'ensemble de test
                predictions = model.predict(X_test)

                # Calcule les métriques d'évaluation
                accuracy = accuracy_score(y_test, predictions)
                precision = precision_score(y_test, predictions)
                recall = recall_score(y_test, predictions)
                f1 = f1_score(y_test, predictions)

                # Affiche les métriques
                print("Model: {}".format(model_name))
                print("Vectorizer Type: {}".format(vectorizer_type))
                print("Accuracy: {:.2f}".format(accuracy))
                print("Precision: {:.2f}".format(precision))
                print("Recall: {:.2f}".format(recall))
                print("F1 Score: {:.2f}".format(f1))
                print("Confusion Matrix:\n", confusion_matrix(y_test, predictions))

    else:
        print("Evaluation for the specified task is not implemented yet")

# Ajouter les commandes à la ligne de commande
cli.add_command(train)
cli.add_command(predict)
cli.add_command(evaluate)

if __name__ == "__main__":
    cli()
