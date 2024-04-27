import sys
import yaml
import pandas as pd
import random
from datetime import datetime
import csv
import uuid

def get_config_data(path):
    data = None
    with open(path, "r") as file:
        data = yaml.safe_load(file)
    return data

def classify_dataset(dataset, config):
    neutral_text = []
    medical_condition_text = []
    feature_tags = config["feature_presence_tags"]

    for _, row in dataset.iterrows():
        feature = row[config["feature_column"]]
        if not isinstance(feature, (int, float, complex)):
            feature = feature.strip()
        
        if feature in feature_tags:
            medical_condition_text.append(row[config["text_column"]])
        else:
            neutral_text.append(row[config["text_column"]])

    return neutral_text, medical_condition_text

def get_data_by_percentage(config, neutral_text, medical_condition_text):
    neutral_size = int(len(neutral_text) * config["proportion_neutral"])
    medical_condition_size = int(len(medical_condition_text) * config["proportion_with_feature"])

    random_neutral = random.sample(neutral_text, neutral_size)
    random_medical_condition = random.sample(medical_condition_text, medical_condition_size)

    return random_neutral, random_medical_condition

def save(config, random_neutral, random_medical_condition, final_dataset_name):
    with open(final_dataset_name, "a") as tsv_file:
        tsv_writer = csv.writer(tsv_file, delimiter='\t', lineterminator='\n')
        save_category(random_neutral, 4, tsv_writer)
        save_category(random_medical_condition, config["category"], tsv_writer)

def save_category(text_list, category, tsv_writer):
    for text in text_list:
        tsv_writer.writerow([uuid.uuid4(), text, category])

def generate_dataset(config_datasets, final_dataset_name):
    random_neutral = []
    random_medical_condition = []
    for idx in range(1, len(config_datasets["datasets"]) + 1):
        config = config_datasets["datasets"][f"dataset{idx}"]
        dataset = pd.read_csv(config["url"])
        neutral_text, medical_condition_text = classify_dataset(dataset, config)
        random_neutral, random_medical_condition = get_data_by_percentage(config, neutral_text, medical_condition_text)
        save(config, random_neutral, random_medical_condition, final_dataset_name)

def initialize_dataset():
    utc_datetime = datetime.utcnow()
    utc_datetime_str = utc_datetime.strftime("%Y-%m-%d_%H-%M-%S")
    headers = ['id', 'text', 'category']
    file_name = f"./output__{utc_datetime_str}.tsv"

    with open(file_name, "a") as tsv_file:
        tsv_writer = csv.writer(tsv_file, delimiter='\t', lineterminator='\n')
        tsv_writer.writerow(headers)

    return file_name

def main():
    path_config = sys.argv[1]
    final_dataset_name = initialize_dataset()
    config_datasets = get_config_data(path_config)
    generate_dataset(config_datasets, final_dataset_name)

if __name__ == "__main__":
    sys.exit(main())