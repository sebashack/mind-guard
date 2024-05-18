import ast
from datetime import datetime
from pandarallel import pandarallel
import argparse
import os
import pandas as pd
import spacy
import sys


def tokenize_text(text, spacy_nlp):
    removal = [
        "ADV",
        "PRON",
        "CCONJ",
        "PUNCT",
        "PART",
        "DET",
        "ADP",
        "SPACE",
        "NUM",
        "SYM",
    ]
    doc = spacy_nlp(text.lower())
    return [
        token.lemma_
        for token in doc
        if token.pos_ not in removal and not token.is_stop and token.is_alpha
    ]

def read_dataset_with_tokens(tsv_path):
    df = pd.read_csv(tsv_path, sep="\t")
    df["tokens"] = df["tokens"].apply(ast.literal_eval)

    return df


def main():
    parser = argparse.ArgumentParser(description="Generate tokenized dataset")
    parser.add_argument(
        "-d", "--dataset", help="path to TSV dataset", required=True, type=str
    )

    args = parser.parse_args()

    andarallel.initialize(progress_bar=True)

    df = pd.read_csv(args.dataset, sep="\t")
    spacy_nlp = spacy.load("en_core_web_sm")

    if "text" not in df.columns:
        raise ValueError("The input TSV file must contain a 'text' column.")

    df["tokens"] = df["text"].parallel_apply(lambda x: tokenize_text(x, spacy_nlp))

    utc_datetime = datetime.utcnow()
    utc_datetime_str = utc_datetime.strftime("%Y-%m-%d_%H-%M-%S")

    output_path = os.path.join(
        os.getcwd(), f"dataset-with-tokens-{utc_datetime_str}.tsv"
    )

    df.to_csv(output_path, sep="\t", index=False)


if __name__ == "__main__":
    sys.exit(main())
