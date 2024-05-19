import matplotlib.pyplot as plt
import numpy as np
import os
import ast
import sys
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans


def read_dataset_with_tokens(tsv_url):
    df = pd.read_csv(tsv_url, sep="\t")
    df["tokens"] = df["tokens"].apply(ast.literal_eval)
    df["tokens_str"] = df["tokens"].apply(lambda x: " ".join(x))

    return df


def get_top_n_words_per_cluster(df, labels, words, N=10):
    df["cluster"] = labels

    cluster_tfidf = df.groupby("cluster").sum()

    top_words = {}
    for cluster in cluster_tfidf.index:
        top_n_words = cluster_tfidf.loc[cluster].nlargest(N).index.tolist()
        top_words[cluster] = top_n_words

    return top_words


def find_optimal_k(df, n_components, max_k=10):
    tfidf = TfidfVectorizer()
    csr_mat = tfidf.fit_transform(df["tokens_str"])
    words = tfidf.get_feature_names_out()

    svd = TruncatedSVD(n_components=n_components)
    reduced_data = svd.fit_transform(csr_mat)

    inertia = []
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(reduced_data)
        inertia.append(kmeans.inertia_)

    plt.figure(figsize=(10, 5))
    plt.plot(range(2, max_k + 1), inertia, "bx-")
    plt.xlabel("k")
    plt.ylabel("Inertia")
    plt.title("Elbow Method For Optimal k")

    plt.xticks(range(2, max_k + 1, 2))
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def plot_explained_variance(df, max_components=500):
    tfidf = TfidfVectorizer()
    csr_mat = tfidf.fit_transform(df["tokens_str"])

    svd = TruncatedSVD(n_components=max_components)
    svd.fit(csr_mat)

    explained_variance = np.cumsum(svd.explained_variance_ratio_)

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, max_components + 1), explained_variance, marker="o")
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("Explained Variance by Number of Components")

    plt.xticks(range(1, max_components, 100))
    plt.grid(True)

    plt.show()


def pipeline(df, n_components, n_clusters):
    tfidf = TfidfVectorizer()
    csr_mat = tfidf.fit_transform(df["tokens_str"])
    words = tfidf.get_feature_names_out()
    print(f"Original dimensionality = {csr_mat.shape}")

    svd = TruncatedSVD(n_components=n_components)
    kmeans = KMeans(n_clusters=n_clusters)
    pipeline = make_pipeline(svd, kmeans)

    pipeline.fit(csr_mat)
    labels = pipeline.predict(csr_mat)

    n_words = 15
    top_words = get_top_n_words_per_cluster(
        pd.DataFrame(csr_mat.toarray(), columns=words), labels, words, N=n_words
    )

    print(f"Top {n_words} most relevant words per cluster: ")
    for cluster_id, words in top_words.items():
        print(f"Cluster {cluster_id}:")
        for w in words:
            print(f"- {w}")


def main():
    depression_url = "https://mindguard.s3.amazonaws.com/refined/depression-with-tokens/depression-with-tokens.tsv"
    suicide_url = "https://mindguard.s3.amazonaws.com/refined/suicide-with-tokens/suicide-with-tokens.tsv"
    cyberbullying_url = "https://mindguard.s3.amazonaws.com/refined/cyberbullying-with-tokens/cyberbullying-with-tokens.tsv"

    print("-- Depression data analysis --")
    df_depression = read_dataset_with_tokens(depression_url)

    # plot_explained_variance(df_depression, max_components=2000) # Run this analysis to visualize optimal number of components
    # find_optimal_k(df_depression, n_components=1800, max_k=100) # Run this anlysis to visualize optimal k
    pipeline(df_depression, n_components=1800, n_clusters=40)

    # print("-- Suicide data analysis --")

    # df_suicide = read_dataset_with_tokens(suicide_url)
    # pipeline(df_suicide, n_components=100, n_clusters=5)

    # print("-- Cyberbullying data analysis --")
    # df_cyberbullying = read_dataset_with_tokens(cyberbullying_url)
    # pipeline(df_cyberbullying, n_components=100, n_clusters=5)


if __name__ == "__main__":
    sys.exit(main())
