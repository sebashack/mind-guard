import sys
import torch
from transformers import LlamaTokenizer, LlamaForCausalLM
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training

eval_prompt = """
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

    # prepare int-8 model for training
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model


def main():
    if torch.cuda.is_available():
        print("CUDA Version:", torch.version.cuda)
    else:
        raise Exception("CUDA is not available")

    llama_model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    fined_tuned_model_id = "./training_results__19-04-2024__20-32-10"
    tokenizer = LlamaTokenizer.from_pretrained(llama_model_id)

    model = LlamaForCausalLM.from_pretrained(
        fined_tuned_model_id,
        load_in_8bit=True,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    model = prepare_model(model)
    model_input = tokenizer(eval_prompt, return_tensors="pt").to("cuda")
    model.eval()

    with torch.no_grad():
        print(
            tokenizer.decode(
                model.generate(**model_input, max_new_tokens=100)[0],
                skip_special_tokens=True,
            )
        )


if __name__ == "__main__":
    sys.exit(main())
