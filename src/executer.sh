for vectorizer_type in count hashing_vectorizer tfidf_vectorizer
    do
        python3 src/main.py train --task=is_comic_video --input_file=src/data/raw/train.csv --model_dump=src/model/dump_${vectorizer_type}.json --vectorizer_type=${vectorizer_type}
        python3 src/main.py evaluate --task=is_comic_video --input_file=src/data/raw/train.csv --vectorizer_type=${vectorizer_type}
    done
