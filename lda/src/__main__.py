from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import ast
import sys
import pandas as pd
from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaMulticore
from gensim.models import CoherenceModel
import pyLDAvis.gensim_models


def read_dataset_with_tokens(tsv_url):
    df = pd.read_csv(tsv_url, sep="\t")
    df["tokens"] = df["tokens"].apply(ast.literal_eval)
    df["tokens_str"] = df["tokens"].apply(lambda x: " ".join(x))

    return df


def get_dictionary_from_tokens(tokens, no_below, no_above, keep_n):
    dictionary = Dictionary(tokens)
    dictionary.filter_extremes(no_below=no_below, no_above=no_above, keep_n=keep_n)

    return dictionary


def get_coherence(lda_model, corpus, dictionary, tokens):
    cm = CoherenceModel(
        model=lda_model,
        texts=tokens,
        corpus=corpus,
        dictionary=dictionary,
        coherence="c_v",
    )

    return cm.get_coherence()


def find_optimal_num_topics(
    df, no_below, no_above, keep_n, workers, lda_iters, passes, n_topics, step=3
):
    tokens = df["tokens"].tolist()
    dictionary = get_dictionary_from_tokens(tokens, no_below, no_above, keep_n)
    corpus = [dictionary.doc2bow(doc) for doc in tokens]

    coherence_values_c_v = []
    model_list = []

    for num_topics in range(2, n_topics, step):
        lda_model = LdaMulticore(
            corpus=corpus,
            id2word=dictionary,
            iterations=lda_iters,
            num_topics=num_topics,
            workers=workers,
            passes=passes,
        )
        model_list.append(lda_model)

        coherence_c_v = get_coherence(lda_model, corpus, dictionary, tokens)
        coherence_values_c_v.append(coherence_c_v)

        print(f"Num Topics: {num_topics}, Coherence Coherence C_V: {coherence_c_v}")

    # Plot coherence values
    x = range(2, n_topics, step)
    plt.figure(figsize=(14, 7))

    plt.plot(x, coherence_values_c_v, marker="o")
    plt.xlabel("Number of Topics")
    plt.ylabel("Coherence Score (C_V)")
    plt.title("Optimal Number of Topics (C_V)")

    plt.tight_layout()
    plt.show()

    return model_list, coherence_values_c_v


def pipeline(df, no_below, no_above, keep_n, workers, lda_iters, passes, num_topics):
    tokens = df["tokens"].tolist()

    dictionary = get_dictionary_from_tokens(tokens, no_below, no_above, keep_n)

    corpus = [dictionary.doc2bow(doc) for doc in tokens]

    lda_model = LdaMulticore(
        corpus=corpus,
        id2word=dictionary,
        iterations=lda_iters,
        num_topics=num_topics,
        workers=workers,
        passes=passes,
    )

    coherence_c_v = get_coherence(lda_model, corpus, dictionary, tokens)
    print("Coherence_c_v: ", coherence_c_v)

    utc_datetime = datetime.utcnow()
    utc_datetime_str = utc_datetime.strftime("%Y-%m-%d_%H-%M-%S")

    lda_display = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)
    pyLDAvis.save_html(
        lda_display, os.path.join(os.getcwd(), f"lda_report__{utc_datetime_str}.html")
    )


def main():
    cyberbullying_url = "https://mindguard.s3.amazonaws.com/refined/cyberbullying-with-tokens/cyberbullying-with-tokens.tsv"

    print("-- Cyberbullying data analysis --")
    df_cyberbullying = read_dataset_with_tokens(cyberbullying_url)
    # find_optimal_num_topics(
    #    df_cyberbullying,
    #    no_below=10,
    #    no_above=0.5,
    #    keep_n=100,
    #    workers=13,
    #    lda_iters=80,
    #    passes=10,
    #    n_topics=15,
    #    step=1,
    # )

    pipeline(
        df_cyberbullying,
        no_below=10,
        no_above=0.5,
        keep_n=100,
        workers=13,
        lda_iters=80,
        passes=10,
        num_topics=6,
    )


if __name__ == "__main__":
    sys.exit(main())
