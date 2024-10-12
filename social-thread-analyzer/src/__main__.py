import json
import argparse
import os
import random
import requests
import subprocess
import sys
import time
import torch
import pandas as pd

from transformers import AutoTokenizer, pipeline
from llama_recipes.inference.model_utils import load_model, load_peft_model


def read_thread(tsv_path):
    df = pd.read_csv(tsv_path, sep="\t")

    thread = ""
    posts = []
    for index, row in df.iterrows():
        username = row["user"]
        post = row["post"].replace("\n", " ")

        thread += f"{username}: {post}\n"
        posts.append(post)

    thread = f"Summarize this dialog:\n{thread.strip()}\n---\nSummary:\n"

    return thread, posts


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


tiny_llama_fine_tunned_model = "tiny-llama"
distil_bert_fined_tunned_model = "distilbert"

categories = {
    "LABEL_0": "neutral",
    "LABEL_1": "depression_and_anxiety",
    "LABEL_2": "suicidal_ideation",
    "LABEL_3": "cyber_bullying",
}


def llama_summary(model_path, thread):
    base_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    # seed = 555181258  # random.randint(0, 999999999)
    seed = 123098  # random.randint(0, 999999999)
    print(f"SEED = {seed}")
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)

    use_quantization = True
    model = load_model(base_model, use_quantization, use_fast_kernels=False)
    model = load_peft_model(model, model_path)

    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token
    batch = tokenizer(
        thread,
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
            min_length=10,
            use_cache=True,
            top_k=50,
            repetition_penalty=1.0,
            length_penalty=1,
        )

    e2e_inference_time = (time.perf_counter() - start) * 1000
    print(f"the inference time is {e2e_inference_time} ms")
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    summary = output_text.split("Summary:\n")[1].strip()

    return summary


def distil_bert_summary_classification(model_path, summary):
    model = pipeline("text-classification", model=model_path)
    r = model([summary])[0]
    return {"category": categories[r["label"]], "score": r["score"]}


def distil_bert_post_classification(model_path, posts):
    model = pipeline("text-classification", model=model_path)
    results = {}
    for i, r in enumerate(model(posts)):
        results[i] = {"category": categories[r["label"]], "score": r["score"]}

    return results


def main():
    parser = argparse.ArgumentParser(description="Analyze social media post")
    parser.add_argument(
        "-t",
        "--thread",
        required=True,
        type=str,
        help="Path to social thread",
    )

    if torch.cuda.is_available():
        print("CUDA Version:", torch.version.cuda)
    else:
        raise Exception("CUDA is not available")

    args = parser.parse_args()

    thread, posts = read_thread(args.thread)

    models_dir = os.path.join(os.getcwd(), "_models")

    ft_tiny_llama_path = os.path.join(models_dir, tiny_llama_fine_tunned_model)
    ft_distil_bert_path = os.path.join(models_dir, distil_bert_fined_tunned_model)

    summary = llama_summary(ft_tiny_llama_path, thread)

    summary_classification = distil_bert_summary_classification(
        ft_distil_bert_path, summary
    )

    post_classification = distil_bert_post_classification(ft_distil_bert_path, posts)

    result = {
        "summary": summary,
        "summary_classification": summary_classification,
        "post_classification": post_classification,
    }

    with open(os.path.join(os.getcwd(), "classification.json"), "w") as f:
        json.dump(result, f, indent=4)


if __name__ == "__main__":
    sys.exit(main())
