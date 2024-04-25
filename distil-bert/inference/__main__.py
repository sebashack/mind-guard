import torch
import sys
from transformers import pipeline


def main():
    if torch.cuda.is_available():
        print("CUDA Version:", torch.version.cuda)
    else:
        raise Exception("CUDA is not available")

    model_dirpath = "./fine_tuned_distil_bert_model__25-04-2024__20-22-10/checkpoint-1250"
    model = pipeline("text-classification", model=model_dirpath)
    results = model(["im meeting up with one of my besties tonight! Cant wait!! - GIRL TALK!!"])

    print(results)


if __name__ == "__main__":
    sys.exit(main())
