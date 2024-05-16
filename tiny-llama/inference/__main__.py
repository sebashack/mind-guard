import time
import argparse
import sys
import torch
import random

from transformers import AutoTokenizer
from llama_recipes.inference.model_utils import load_model, load_peft_model


def prepare_input(dialog):
    return f"Summarize this dialog:\n{dialog.strip()}\n---\nSummary:\n"


def read_dialog(file_path):
    with open(file_path, "r") as file:
        dialog = file.read()
    return prepare_input(dialog)


def prepare_model(model):
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
    )

    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Run inferences on pretrained llama models"
    )
    parser.add_argument(
        "-m",
        "--model",
        required=False,
        type=str,
        help="Path to directory with model. If this argument is not provided, the original  TinyLlama model will be loaded",
    )
    parser.add_argument(
        "-d",
        "--dialog",
        required=True,
        type=str,
        help="Path to file with dialog to summarize",
    )

    if torch.cuda.is_available():
        print("CUDA Version:", torch.version.cuda)
    else:
        raise Exception("CUDA is not available")

    args = parser.parse_args()

    base_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    fine_tuned_peft_model = args.model

    seed = random.randint(0, 999999999)
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)

    use_quantization = True
    model = load_model(base_model, use_quantization)

    if fine_tuned_peft_model is not None:
        model = load_peft_model(model, fine_tuned_peft_model)

    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token
    batch = tokenizer(
        read_dialog(args.dialog),
        padding=True,
        truncation=True,
        max_length=None,
        return_tensors="pt",
    )
    batch = {k: v.to("cuda") for k, v in batch.items()}

    start = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            **batch,
            max_new_tokens=250,
            do_sample=True,
            top_p=1.0,
            temperature=1.0,
            min_length=None,
            use_cache=True,
            top_k=50,
            repetition_penalty=1.0,
            length_penalty=1,
        )

    e2e_inference_time = (time.perf_counter() - start) * 1000
    print(f"the inference time is {e2e_inference_time} ms")
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    summary = output_text.split("Summary:\n")[1].strip()

    print(summary)


if __name__ == "__main__":
    sys.exit(main())
