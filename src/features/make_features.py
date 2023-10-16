from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer, TfidfVectorizer
from scipy.sparse import issparse
import pandas as pd

def make_features(df, vectorizer_type):
    if vectorizer_type == "count":
        vectorizer = CountVectorizer()
    elif vectorizer_type == "hashing_vectorizer":
        vectorizer = HashingVectorizer()
    elif vectorizer_type == "tfidf_vectorizer":
        vectorizer = TfidfVectorizer()
    else:
        raise ValueError("Invalid vectorizer_type")
    
    if issparse(df):
        # Si la variable df est une matrice creuse, elle ne possède pas de colonnes.
        # Vous devez gérer le cas spécifique où df est déjà une matrice creuse.
        X = df
        y = None  # Initialisez y à None car il n'y a pas de cible
    else:
        # Sinon, c'est supposé être un DataFrame avec des colonnes.
        if "is_comic" in df.columns:
            y = df["is_comic"].values
        else:
            raise ValueError("Column 'is_comic' not found in the DataFrame.")
        
        X = df["video_name"].astype(str)
        X = vectorizer.fit_transform(X)

    return X, y