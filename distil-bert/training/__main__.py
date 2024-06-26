import argparse
import random
from datetime import datetime
import numpy as np
import os
import sys
import torch
from datasets import Dataset, load_metric
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import classification_report


def compute_metrics(eval_pred):
    load_accuracy = load_metric("accuracy", trust_remote_code=True)
    load_f1 = load_metric("f1", average="weighted", trust_remote_code=True)

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = load_accuracy.compute(predictions=predictions, references=labels)[
        "accuracy"
    ]
    f1 = load_f1.compute(
        predictions=predictions, references=labels, average="weighted"
    )["f1"]

    report = classification_report(labels, predictions, output_dict=True)
    per_category_accuracy = {
        f"accuracy_{category}": report[category]["precision"]
        for category in report
        if category not in ["accuracy", "macro avg", "weighted avg"]
    }
    per_category_f1 = {
        f"f1_{category}": report[category]["f1-score"]
        for category in report
        if category not in ["accuracy", "macro avg", "weighted avg"]
    }

    return {"accuracy": accuracy, "f1": f1, **per_category_accuracy, **per_category_f1}


def main():
    parser = argparse.ArgumentParser(description="Train DISTIL-BERT")
    parser.add_argument(
        "-d",
        "--dataset",
        required=True,
        type=str,
        help="Path TSV file with dataset",
    )

    if torch.cuda.is_available():
        print("CUDA Version:", torch.version.cuda)
    else:
        raise Exception("CUDA is not available")

    args = parser.parse_args()

    seed = random.randint(0, 999999999)
    dataset_path = args.dataset
    dataset = Dataset.from_csv(dataset_path, delimiter="\t")
    dataset = dataset.train_test_split(test_size=0.15, seed=seed)

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def preprocess_function(examples):
        tokenized_inputs = tokenizer(
            examples["text"], truncation=True, max_length=512, padding="max_length"
        )
        tokenized_inputs["labels"] = [int(label) for label in examples["category"]]
        return tokenized_inputs

    tokenized_train = dataset["train"].map(preprocess_function, batched=True)
    tokenized_test = dataset["test"].map(preprocess_function, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=4
    )

    now = datetime.now()
    date_str = now.strftime("%d-%m-%Y__%H-%M-%S")
    output_dir = f"{os.getcwd()}/fine_tuned_distil_bert_model__{date_str}"

    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5,
        weight_decay=0.01,
        save_strategy="epoch",
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    print("** Evaluation **")
    trainer.evaluate()

    evaluation_metrics = trainer.evaluate()

    metrics_file = os.path.join(output_dir, f"{os.getcwd()}/evaluation_metrics.txt")
    with open(metrics_file, "w") as f:
        for key, value in evaluation_metrics.items():
            f.write(f"{key}: {value}\n")


if __name__ == "__main__":
    sys.exit(main())
