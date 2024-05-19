from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaMulticore
from gensim.models import CoherenceModel

from tf_idf import (
    get_tokenazided_df,
)


def topic_modelling_pipeline(
    no_below, no_above, keep_n, workers, lda_iters, passes, num_topics
):
    tokenized_df = get_tokenazided_df()
    tokens = tokenized_to_list(tokenized_df)
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

    coherence_u_mass = get_coherence(
        lda_model, corpus, dictionary, "u_mass", tokens)
    coherence_c_v = get_coherence(lda_model, corpus, dictionary, "c_v", tokens)

    print("Coherence_u_mass: ", coherence_u_mass)
    print("Coherence_c_v: ", coherence_c_v)

    return lda_model, corpus, dictionary


def get_dictionary_from_tokens(tokens, no_below, no_above, keep_n):
    dictionary = Dictionary(tokens)
    dictionary.filter_extremes(
        no_below=no_below, no_above=no_above, keep_n=keep_n)

    return dictionary


def tokenized_to_list(tokenized_df):
    return tokenized_df["tokens"]


def get_model_lda_and_k_optimal(
    no_below,
    no_above,
    keep_n,
    workers,
    iterations,
    passes,
    max_topics,
    type_coherence,
):
    tokenized_df = get_tokenazided_df()
    tokens = tokenized_to_list(tokenized_df)
    dictionary = get_dictionary_from_tokens(tokens, no_below, no_above, keep_n)
    corpus = [dictionary.doc2bow(doc) for doc in tokens]

    topics = []
    score = []
    lda_model = None
    for k in range(1, max_topics):
        lda_model = LdaMulticore(
            corpus=corpus,
            id2word=dictionary,
            iterations=iterations,
            num_topics=k,
            workers=workers,
            passes=passes,
            random_state=100,
        )
        coherence = get_coherence(
            lda_model, corpus, dictionary, type_coherence, tokens)
        topics.append(k)
        score.append(coherence)

    idx_max_coherence = score.index(max(score))

    return topics[idx_max_coherence]


def get_coherence(lda_model, corpus, dictionary, type_coherence, tokens):
    cm = None
    if type_coherence == "u_mass":
        cm = CoherenceModel(
            model=lda_model,
            corpus=corpus,
            dictionary=dictionary,
            coherence=type_coherence,
        )
    else:
        cm = CoherenceModel(
            model=lda_model,
            texts=tokens,
            corpus=corpus,
            dictionary=dictionary,
            coherence="c_v",
        )

    return cm.get_coherence()
