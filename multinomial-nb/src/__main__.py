import random
import numpy as np
import os
import ast
import sys
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report


def read_dataset_with_tokens(tsv_url):
    df = pd.read_csv(tsv_url, sep="\t")
    df["tokens"] = df["tokens"].apply(ast.literal_eval)
    df["tokens_str"] = df["tokens"].apply(lambda x: " ".join(x))

    return df


topic_to_int = {
    "neutral": 0,
    "depression_and_anxiety": 1,
    "suicidal_ideation": 2,
    "cyber_bullying": 3,
}


def main():
    url = "https://mindguard.s3.amazonaws.com/refined/distilbert-with-tokens/distilbert-with-tokens.tsv"
    df = read_dataset_with_tokens(url)

    vectorizer = CountVectorizer(analyzer=lambda s: s.split(), dtype="uint8")

    texts = df["tokens_str"].tolist()

    topic_vector = [int(i) for i in df["category"].tolist()]

    df_countvectorizer = vectorizer.fit_transform(texts)

    seed = random.randint(0, 999999999)

    X_train, X_test, y_train, y_test = train_test_split(
        df_countvectorizer, topic_vector, test_size=0.1, random_state=seed
    )

    classifier = MultinomialNB()
    classifier.fit(X_train, y_train)

    pred = classifier.predict(X_test)

    print(classification_report(y_test, pred))


if __name__ == "__main__":
    sys.exit(main())
