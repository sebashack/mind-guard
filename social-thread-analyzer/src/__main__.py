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


def download_thread(url, output_dir, filename):
    os.makedirs(output_dir, exist_ok=True)

    file_path = os.path.join(output_dir, filename)

    response = requests.get(url)

    response.raise_for_status()

    with open(file_path, "w") as file:
        file.write(response.text)


def download_and_extract_tar_lz(url, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    tar_lz_path = os.path.join(output_dir, "artifact.tar.lz")

    response = requests.get(url, stream=True)
    response.raise_for_status()

    with open(tar_lz_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    subprocess.run(["lzip", "-d", tar_lz_path], check=True)

    tar_path = tar_lz_path[:-3]

    subprocess.run(["tar", "-xf", tar_path, "-C", output_dir], check=True)

    os.remove(tar_path)


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


tiny_llama_fine_tunned_model = "fine_tuned_peft_model__15-05-2024__18-41-12"
tiny_llama_url = "https://mindguard.s3.amazonaws.com/trusted/tiny-llama/002__tiny_llama_fine_tuned_peft_model_5_epochs_15-05-2024__18-41-12.tar.lz"

distil_bert_fined_tunned_model = "distilbert-fine-tunned"
distil_bert_url = "https://mindguard.s3.amazonaws.com/trusted/distil-bert/fine_tuned_distil_bert_model__metrics_5_epochs__28-05-2024__18-38-10.tar.lz"

categories = {
    "LABEL_0": "neutral",
    "LABEL_1": "depression_and_anxiety",
    "LABEL_2": "suicidal_ideation",
    "LABEL_3": "cyber_bullying",
}


def llama_summary(model_path, thread):
    base_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    seed = 555181259  # random.randint(0, 999999999)
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
        "-u",
        "--url",
        required=True,
        type=str,
        help="Path to file with dialog to summarize",
    )

    if torch.cuda.is_available():
        print("CUDA Version:", torch.version.cuda)
    else:
        raise Exception("CUDA is not available")

    args = parser.parse_args()

    output_dir = os.path.join(os.getcwd(), "_tmp")
    download_thread(args.url, output_dir, "thread.tsv")

    thread, posts = read_thread(os.path.join(output_dir, "thread.tsv"))

    models_dir = os.path.join(os.getcwd(), "_models")
    if not os.path.exists(models_dir):
        print("Downloading fined-tunned models for distilbert and tiny-llama... next time this won't be necessary...")
        download_and_extract_tar_lz(tiny_llama_url, models_dir)
        download_and_extract_tar_lz(
            distil_bert_url, os.path.join(models_dir, distil_bert_fined_tunned_model)
        )

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
