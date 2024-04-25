from datetime import datetime
import numpy as np
import os
import sys
import torch
from datasets import load_dataset, load_metric
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)


def compute_metrics(eval_pred):
    load_accuracy = load_metric("accuracy")
    load_f1 = load_metric("f1", average="weighted")

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = load_accuracy.compute(predictions=predictions, references=labels)[
        "accuracy"
    ]
    f1 = load_f1.compute(
        predictions=predictions, references=labels, average="weighted"
    )["f1"]
    return {"accuracy": accuracy, "f1": f1}


def main():
    if torch.cuda.is_available():
        print("CUDA Version:", torch.version.cuda)
    else:
        raise Exception("CUDA is not available")

    dataset_name = "sentiment140"
    dataset = load_dataset(dataset_name)
    small_train_dataset = (
        dataset["train"].shuffle(seed=42).select([i for i in list(range(10000))])
    )
    small_test_dataset = (
        dataset["test"].shuffle(seed=42).select([i for i in list(range(450))])
    )

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def preprocess_function(examples):
        # Remap labels
        label_map = {0: 0, 2: 1, 4: 2}
        labels = [label_map[label] for label in examples["sentiment"]]

        # Tokenize text
        tokenized_inputs = tokenizer(examples["text"], truncation=True)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    tokenized_train = small_train_dataset.map(preprocess_function, batched=True)
    tokenized_test = small_test_dataset.map(preprocess_function, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=3
    )

    now = datetime.now()
    date_str = now.strftime("%d-%m-%Y__%H-%M-%S")
    output_dir = f"{os.getcwd()}/fine_tuned_distil_bert_model__{date_str}"

    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=2,
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
    trainer.evaluate()


if __name__ == "__main__":
    sys.exit(main())
