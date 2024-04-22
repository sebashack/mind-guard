import sys
import torch
from llama_recipes.finetuning import main
from llama_recipes.configs.training import train_config
import random
from dataclasses import asdict


def train():
    if torch.cuda.is_available():
        print("CUDA Version:", torch.version.cuda)
    else:
        raise Exception("CUDA is not available")

    training_config = train_config()

    training_config.model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    training_config.output_dir = "./results"
    training_config.peft_method = "lora"
    training_config.use_peft = True
    training_config.quantization = True
    training_config.use_fp16 = True
    training_config.seed = random.randint(0, 999999999)
    training_config.run_validation = True
    training_config.batch_size_training = 4
    training_config.val_batch_size = 2
    training_config.gradient_accumulation_steps = 1
    training_config.mixed_precision = True
    training_config.one_gpu = True
    training_config.save_model = True
    training_config.batching_strategy = "packing"
    training_config.context_length = 4096
    training_config.save_metrics = True
    training_config.num_epochs = 1
    training_config.num_workers_dataloader = 4
    training_config.dataset = "samsum_dataset"

    main(**asdict(training_config))


if __name__ == "__main__":
    sys.exit(train())
