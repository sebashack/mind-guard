import time
import argparse
import sys
import torch
import random

from transformers import AutoTokenizer
from llama_recipes.inference.model_utils import load_model, load_peft_model

user_prompt = """
Summarize this dialog:
A: Hi Tom, are you busy tomorrow’s afternoon?
B: I’m pretty sure I am. What’s up?
A: Can you go with me to the animal shelter?.
B: What do you want to do?
A: I want to get a puppy for my son.
B: That will make him so happy.
A: Yeah, we’ve discussed it many times. I think he’s ready now.
B: That’s good. Raising a dog is a tough issue. Like having a baby ;-)
A: I'll get him one of those little dogs.
B: One that won't grow up too big;-)
A: And eat too much;-))
B: Do you know which one he would like?
A: Oh, yes, I took him there last Monday. He showed me one that he really liked.
B: I bet you had to drag him away.
A: He wanted to take it home right away ;-).
B: I wonder what he'll name it.
A: He said he’d name it after his dead hamster – Lemmy  - he's  a great Motorhead fan :-)))
---
Summary:
"""


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
    # parser = argparse.ArgumentParser(
    #     description="Run inferences on pretrained llama models"
    # )
    # parser.add_argument(
    #     "-m",
    #     "--model",
    #     required=False,
    #     type=str,
    #     help="Path to directory with model. If this argument is not provided, the original  TinyLlama model will be loaded",
    # )

    if torch.cuda.is_available():
        print("CUDA Version:", torch.version.cuda)
    else:
        raise Exception("CUDA is not available")

    # args = parser.parse_args()

    base_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    fine_tuned_peft_model = "./results"

    seed = random.randint(0, 999999999)
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)

    use_quantization = True
    model = load_model(base_model, use_quantization)
    model = load_peft_model(model, fine_tuned_peft_model)

    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token
    batch = tokenizer(
        user_prompt,
        padding="max_length",
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

    print(output_text)


if __name__ == "__main__":
    sys.exit(main())
