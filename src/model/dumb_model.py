import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from src.features.make_features import make_features

class DumbModel:
    """Dumb model always predicts 0"""
    def __init__(self, model_name, vectorizer_type):
        self.df = None
        self.vectorizer = None
        self.model = None
        self.vectorizer_type = vectorizer_type
        self.model_name = model_name
        self.available_models = {
            "LogisticRegression": LogisticRegression(),
            "MultinomialNB": MultinomialNB(),
            "RandomForest": RandomForestClassifier()
        }
        self.X_train = None
        self.y_train = None

    def train(self, df):
        # Utilisation du même vecteur que dans make_features.py
        self.df = df
        self.X_train, self.y_train = make_features(df, self.vectorizer_type)
        return self.X_train, self.y_train

    def fit(self):
        if self.X_train is not None and self.y_train is not None:
            self.model = self.available_models[self.model_name]
            self.model.fit(self.X_train, self.y_train)
            print("fitted")
        else:
            raise ValueError("Training data not set. Call train() before fit()")

    def predict(self, df):
        if self.vectorizer is not None:
            X, _ = make_features(df, self.vectorizer_type)
            X = X.toarray()
        else:
            X, _ = make_features(df, self.vectorizer_type)
            X = X.toarray()
        if self.model is not None:
            return self.model.predict(X)
        else:
            raise ValueError("Model not trained")

    def dump(self, filename_output):
        if self.model is not None:
            model_data = {
                "model": self.model,
                "vectorizer_type": self.vectorizer_type
            }
            joblib.dump(model_data, filename_output)

    def load(self, filename_input):
        try:
            model_data = joblib.load(filename_input)
            self.model = model_data["model"]
            self.vectorizer_type = model_data["vectorizer_type"]
        except FileNotFoundError:
            raise ValueError("Model file not found")