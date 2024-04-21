from datetime import datetime
import os
import sys
import torch
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    default_data_collator,
    Trainer,
    TrainingArguments,
)
from llama_recipes.utils.dataset_utils import get_preprocessed_dataset
from llama_recipes.configs.datasets import samsum_dataset
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    prepare_model_for_kbit_training,
)
from transformers import TrainerCallback
from contextlib import nullcontext


def create_peft_config(model):
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
    )

    # prepare int-8 model for training
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model, peft_config


def main():
    if torch.cuda.is_available():
        print("CUDA Version:", torch.version.cuda)
    else:
        raise Exception("CUDA is not available")

    now = datetime.now()
    date_str = now.strftime("%d-%m-%Y__%H-%M-%S")

    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    output_dir = f"./training_results__{date_str}"

    tokenizer = LlamaTokenizer.from_pretrained(model_id)

    model = LlamaForCausalLM.from_pretrained(
        model_id, load_in_8bit=True, device_map="auto", torch_dtype=torch.float16
    )

    train_dataset = get_preprocessed_dataset(tokenizer, samsum_dataset, "train")
    model, lora_config = create_peft_config(model)

    profiler = nullcontext()

    config = {
        "lora_config": lora_config,
        "learning_rate": 1e-4,
        "num_train_epochs": 2,
        "gradient_accumulation_steps": 2,
        "per_device_train_batch_size": 2,
        "gradient_checkpointing": False,
    }

    # Define training args
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        bf16=True,  # Use BF16 if available
        # logging strategies
        logging_dir=f"{output_dir}/logs",
        logging_strategy="steps",
        logging_steps=10,
        save_strategy="epoch",
        optim="adamw_torch_fused",
        max_steps=-1,
        **{k: v for k, v in config.items() if k != "lora_config"},
    )

    with profiler:
        # Create Trainer instance
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=default_data_collator,
            callbacks=[],
        )

        # Start training
        trainer.train()

    model.save_pretrained(output_dir)

if __name__ == "__main__":
    sys.exit(main())
