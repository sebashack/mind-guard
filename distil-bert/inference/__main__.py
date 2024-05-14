import argparse
import torch
import sys
from transformers import pipeline

categories = {
    "LABEL_0": "neutral",
    "LABEL_1": "depression_and_anxiety",
    "LABEL_2": "suicidal_ideation",
    "LABEL_3": "cyber_bullying",
}


def main():
    parser = argparse.ArgumentParser(description="Inferences with DISTIL-BERT")
    parser.add_argument(
        "-m",
        "--model",
        required=True,
        type=str,
        help="Path model directory",
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        type=str,
        help="Path to input text file",
    )

    if torch.cuda.is_available():
        print("CUDA Version:", torch.version.cuda)
    else:
        raise Exception("CUDA is not available")

    args = parser.parse_args()

    model = pipeline("text-classification", model=args.model)

    with open(args.input, "r") as file:
        texts = file.read()
        inputs = [s.strip() for s in texts.split("\n>>\n")]
        for r in model(inputs):
            print(f"{categories[r['label']]}: {r['score']}")


if __name__ == "__main__":
    sys.exit(main())
