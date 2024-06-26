import argparse
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

MODEL_NAME = "google/gemma-2b"
MAX_NEW_TOKENS = 20


parser = argparse.ArgumentParser(description="Local Gemma 2")
parser.add_argument(
    "--model-name",
    default=MODEL_NAME,
    type=str,
    help="",
)
parser.add_argument(
    "--max-new-tokens",
    default=MAX_NEW_TOKENS,
    type=int,
    help="",
)


def main():
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, device_map="auto", torch_dtype=torch.bfloat16)

    model_inputs = tokenizer("Hello world.", return_tensors="pt").to(model.device)
    streamer = TextStreamer(tokenizer, {"skip_special_tokens": True})

    _ = model.generate(
        **model_inputs,
        max_length=args.max_new_tokens,
        streamer=streamer,
        do_sample=True,
    )


if __name__ == '__main__':
    main()
