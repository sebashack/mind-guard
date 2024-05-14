import random
import math
import sys
import yaml
import pandas as pd
from datetime import datetime
import json
import argparse
import os

json_categories = {"0": "neutral"}
json_metadata = {}


def save_to_dataframe(text_list, category, df):
    data = pd.DataFrame({"text": text_list, "category": [category] * len(text_list)})
    return pd.concat([df, data], ignore_index=True)


def save_classification(config, neutral_texts, featured_texts, df):
    df = save_to_dataframe(neutral_texts, 0, df)
    df = save_to_dataframe(featured_texts, config["category"], df)

    return df


def count_words(text):
    return len(text.split())


def split_dataset(dataset, max_word_length, config):
    neutral_texts = []
    featured_texts = []
    feature_tags = [s.strip() for s in config["feature_presence_tags"]]

    discarded = 0
    for _, row in dataset.iterrows():
        feature = row[config["feature_column"]]
        if isinstance(feature, int) or isinstance(feature, float):
            if math.isnan(feature):
                continue
            feature = str(int(feature))

        feature = feature.strip()

        txt = row[config["text_column"]]

        if not isinstance(txt, str) or txt is None or txt.isspace():
            continue

        txt = txt.replace("\t", " ")

        if count_words(txt) > max_word_length:
            discarded += 1
            continue

        if feature in feature_tags:
            featured_texts.append(txt)
        else:
            neutral_texts.append(txt)

    if discarded > 0:
        print(f"{discarded} text(s) exceeded max word size {max_word_length}")

    random.shuffle(neutral_texts)
    random.shuffle(featured_texts)

    return neutral_texts, featured_texts


def save_dataset(df, counts):
    utc_datetime = datetime.utcnow()
    utc_datetime_str = utc_datetime.strftime("%Y-%m-%d_%H-%M-%S")
    json_obj = {
        "categories": json_categories,
        "metadata": json_metadata,
        "counts": counts,
    }

    output_dir_path = os.path.join(os.getcwd(), f"output-{utc_datetime_str}")
    os.makedirs(output_dir_path, exist_ok=True)

    with open(os.path.join(output_dir_path, "metadata.json"), "w") as json_file:
        json.dump(json_obj, json_file, indent=4)

    df.to_csv(os.path.join(output_dir_path, "dataset.tsv"), sep="\t", index=False)


def generate_dataset(config_datasets):
    df = pd.DataFrame(columns=["text", "category"])
    max_word_length = config_datasets["max_word_length"]

    counts = {"neutral": 0}

    for dataset_name, config in config_datasets["datasets"].items():
        print(f"Processing dataset: {dataset_name}")
        dataset = pd.read_csv(config["url"], sep="\t")
        neutral_texts, featured_texts = split_dataset(dataset, max_word_length, config)

        neutral_size = round(len(neutral_texts) * config["proportion_neutral"])
        featured_size = round(len(featured_texts) * config["proportion_with_feature"])

        df = save_classification(
            config,
            neutral_texts[:neutral_size],
            featured_texts[:featured_size],
            df,
        )

        label = config["label"]
        metadata = {
            "neutral": neutral_size,
            "featured": featured_size,
            "total_neutral": len(neutral_texts),
            "total_featured": len(featured_texts),
            "label": label,
            "dataset": dataset_name,
            "dataset_url": config["url"],
        }

        json_categories[config["category"]] = label
        json_metadata[dataset_name] = metadata

        counts["neutral"] += neutral_size
        if config["label"] in counts:
            counts[label] += featured_size
        else:
            counts[label] = featured_size

    save_dataset(df, counts)


def create_parser():
    parser = argparse.ArgumentParser(description="Generate general dataset")
    parser.add_argument(
        "-c", help="path to yaml configuration", required=True, type=str
    )

    return parser.parse_args()


def main():
    args = create_parser()

    config_datasets = None
    with open(args.c, "r") as file:
        config_datasets = yaml.safe_load(file)

    generate_dataset(config_datasets)


if __name__ == "__main__":
    sys.exit(main())
