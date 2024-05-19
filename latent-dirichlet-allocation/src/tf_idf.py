import uuid
import ast
import pandas as pd

from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, StringType
from pyspark.ml.feature import CountVectorizer, IDF
from pyspark.ml import Pipeline, PipelineModel


def get_tokenazided_df():
    tsv_url = "https://spulido1lab1.s3.amazonaws.com/mind-guard/tokenized/dataset-with-tokens.tsv"
    pd_df = read_dataset_with_tokens(tsv_url)

    return pd_df


def read_dataset_with_tokens(tsv_url):
    df = pd.read_csv(tsv_url, sep="\t")
    df["tokens"] = df["tokens"].apply(ast.literal_eval)

    return df


def tf_idf_pipeline(spacy_nlp, df):
    tokenized_df = get_tokenazided_df(df, spacy_nlp)

    tokenized_df.select("tokenized_text").show(10, truncate=False)

    model = compute_tf_idf_model(tokenized_df, vocab_size=20)

    model.save(f"./{str(uuid.uuid4())}")

    vocabulary, tfidf_df = model_to_tf_idf(model, tokenized_df)

    print(vocabulary)
    tfidf_df.select("tfidf_features").show(10, truncate=False)


def tf_idf_pipeline_with_saved_model(model_dir_path, spacy_nlp, df):
    tokenized_df = get_tokenazided_df(df, spacy_nlp)

    model = load_tf_idf_model(model_dir_path)

    vocabulary, tfidf_df = model_to_tf_idf(model, tokenized_df)

    print(vocabulary)
    tfidf_df.select("tfidf_features").show(10, truncate=False)


def compute_tf_idf_model(tokenized_df, vocab_size):
    cv = CountVectorizer(
        inputCol="tokenized_text",
        outputCol="tf_features",
        vocabSize=vocab_size,
        minDF=1.0,
    )
    idf = IDF(inputCol="tf_features", outputCol="tfidf_features")

    pipeline = Pipeline(stages=[cv, idf])

    model = pipeline.fit(tokenized_df)

    return model


def model_to_tf_idf(model, tokenized_df):
    tfidf_df = model.transform(tokenized_df)

    vocabulary = model.stages[0].vocabulary

    return vocabulary, tfidf_df


def load_tf_idf_model(path):
    model = PipelineModel.load(path)

    return model
