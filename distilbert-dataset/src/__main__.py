import sys
import yaml
import pandas as pd
import uuid
from datetime import datetime

def get_config_data(path):
    data = None
    with open(path, "r") as file:
        data = yaml.safe_load(file)
    return data

def save_to_dataframe(text_list, category, df):
    for text in text_list:
        data = pd.DataFrame({
            "id": [uuid.uuid4()],
            "text": [text],
            "category": [category]
        })
        df = pd.concat([df, data])

    return df

def save(config, random_neutral, random_medical_condition, df):
    df = save_to_dataframe(random_neutral, 4, df)
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
        medical_condition_text.append(text_column) if feature in feature_tags else neutral_text.append(text_column)

    return neutral_text, medical_condition_text

def generate_dataset(config_datasets, df):
    for idx in range(1, len(config_datasets["datasets"]) + 1):
        config = config_datasets["datasets"][f"dataset{idx}"]
        dataset = pd.read_csv(config["url"])
        neutral_text, medical_condition_text = classify_dataset(dataset, config)

        neutral_size = int(len(neutral_text) * config["proportion_neutral"])
        medical_condition_size = int(len(medical_condition_text) * config["proportion_with_feature"])

        df = save(config, neutral_text[:neutral_size], medical_condition_text[:medical_condition_size], df)
    
    return df

def save_to_tsv(df):
    utc_datetime = datetime.utcnow()
    utc_datetime_str = utc_datetime.strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"./output__{utc_datetime_str}.tsv"
    df.to_csv(file_name, sep='\t', index=False)

def main():
    df = pd.DataFrame(columns=['id', 'text', 'category'])
    path_config = sys.argv[1]
    config_datasets = get_config_data(path_config)
    df = generate_dataset(config_datasets, df)
    save_to_tsv(df)

if __name__ == "__main__":
    sys.exit(main())