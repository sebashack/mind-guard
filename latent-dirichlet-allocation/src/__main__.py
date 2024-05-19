import sys
import os
import pyLDAvis.gensim_models
import ast
from pyspark.sql import SparkSession
import pandas as pd


from datetime import datetime


from topic_modeling import (
    topic_modelling_pipeline
)


def read_dataset_with_tokens(tsv_url):
    df = pd.read_csv(tsv_url, sep="\t")
    df["tokens"] = df["tokens"].apply(ast.literal_eval)

    return df


def main():
    print("Hello world!")
    spark = SparkSession.builder.master("local[*]").getOrCreate()
    spark.sparkContext.addPyFile(f"{os.getcwd()}/src/tf_idf.py")

    lda_model, corpus, dictionary = topic_modelling_pipeline(
        no_below=5,
        no_above=0.5,
        keep_n=50,
        workers=2,
        lda_iters=1,
        passes=10,
        num_topics=10,
    )

    utc_datetime = datetime.utcnow()
    utc_datetime_str = utc_datetime.strftime("%Y-%m-%d_%H-%M-%S")

    lda_display = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)
    pyLDAvis.save_html(lda_display, f"lda_report__{utc_datetime_str}.html")


if __name__ == "__main__":
    sys.exit(main())
