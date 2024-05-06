import sys
import yaml
import pandas as pd
from datetime import datetime
import json
import argparse
import os

json_categories = {"0": "Neutral"}
json_metadata = {}


def get_config_data(path):
    data = None
    with open(path, "r") as file:
        data = yaml.safe_load(file)
    return data


def save_to_dataframe(text_list, category, df):
    data = pd.DataFrame({"text": text_list, "category": [category] * len(text_list)})
    return pd.concat([df, data], ignore_index=True)


def save_classification(config, random_neutral, random_medical_condition, df):
    df = save_to_dataframe(random_neutral, 0, df)
    df = save_to_dataframe(random_medical_condition, config["category"], df)

    return df


def classify_dataset(dataset, config):
    neutral_text = []
    medical_condition_text = []
    feature_tags = config["feature_presence_tags"]

    for _, row in dataset.iterrows():
        feature = row[config["feature_column"]]
        if not isinstance(feature, (int, float, complex)):
            feature = feature.strip()

        text_column = row[config["text_column"]]
        medical_condition_text.append(
            text_column
        ) if feature in feature_tags else neutral_text.append(text_column)

    return neutral_text, medical_condition_text


def save_to_json(json_obj, utc_datetime):
    file_name = f"./metrics/metrics__{utc_datetime}.json"
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, "w") as json_file:
        json.dump(json_obj, json_file, indent=4)


def save_to_tsv(df, utc_datetime):
    file_name = f"./datasets/output__{utc_datetime}.tsv"
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    df.to_csv(file_name, sep="\t", index=False)


def save(df):
    utc_datetime = datetime.utcnow()
    utc_datetime_str = utc_datetime.strftime("%Y-%m-%d_%H-%M-%S")
    json_obj = {"categories": json_categories, "metadata": json_metadata}
    save_to_tsv(df, utc_datetime_str)
    save_to_json(json_obj, utc_datetime_str)


def save_metrics(category, tag, dataset_name, metadata):
    json_categories[category] = tag
    json_metadata[dataset_name] = metadata


def generate_metadata(
    neutral_size, medical_condition_size, neutral_text, medical_condition_text
):
    return {
        "neutral": neutral_size,
        "featured": medical_condition_size,
        "total_neutral": len(neutral_text),
        "total_featured": len(medical_condition_text),
    }


def generate_dataset(config_datasets):
    df = pd.DataFrame(columns=["text", "category"])

    for dataset_name, config in config_datasets["datasets"].items():
        config = config_datasets["datasets"][dataset_name]
        dataset = pd.read_csv(config["url"])
        neutral_text, medical_condition_text = classify_dataset(dataset, config)

        neutral_size = int(len(neutral_text) * config["proportion_neutral"])
        medical_condition_size = int(
            len(medical_condition_text) * config["proportion_with_feature"]
        )

        df = save_classification(
            config,
            neutral_text[:neutral_size],
            medical_condition_text[:medical_condition_size],
            df,
        )

        metadata = generate_metadata(
            neutral_size, medical_condition_size, neutral_text, medical_condition_text
        )
        save_metrics(config["category"], config["label"], dataset_name, metadata)

    save(df)


def create_parser():
    parser = argparse.ArgumentParser(description="Generate general dataset")
    parser.add_argument("-c", help="path yaml configuration", type=str)

    return parser.parse_args()


def main():
    args = create_parser()
    config_datasets = get_config_data(args.c)
    generate_dataset(config_datasets)


if __name__ == "__main__":
    sys.exit(main())
