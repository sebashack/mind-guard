import os
import ast
import sys
import pandas as pd
from pyspark.sql.functions import col, explode
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    ArrayType,
    IntegerType,
    StringType,
    StructType,
    StructField,
)


def read_dataset_with_tokens(tsv_url):
    df = pd.read_csv(tsv_url, sep="\t")
    df["tokens"] = df["tokens"].apply(ast.literal_eval)

    return df


def token_counts_dataframe(df, spark):
    exploded_df = df.withColumn("token", explode(col("tokens")))

    token_counts_df = exploded_df.groupBy("token").count()

    sorted_token_counts_df = token_counts_df.orderBy(col("count").desc())

    return sorted_token_counts_df


def main():
    spark = SparkSession.builder.appName("WordCount").getOrCreate()

    tsv_url = "https://spulido1lab1.s3.amazonaws.com/mind-guard/tokenized/dataset-with-tokens.tsv"

    pd_df = read_dataset_with_tokens(tsv_url)

    df = spark.createDataFrame(pd_df)

    df_depression = df.filter(df.category == 1)
    df_suicide = df.filter(df.category == 2)
    df_cyberbullying = df.filter(df.category == 3)

    token_counts_dataframe(df_depression, spark).show(30)
    token_counts_dataframe(df_suicide, spark).show(30)
    token_counts_dataframe(df_cyberbullying, spark).show(30)


if __name__ == "__main__":
    sys.exit(main())
