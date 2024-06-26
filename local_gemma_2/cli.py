import argparse
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer, set_seed

MODEL_NAME = "google/gemma-1.1-7b-it"
MAX_NEW_TOKENS = 100


parser = argparse.ArgumentParser(description="Local Gemma 2")
parser.add_argument(
    "--model-name",
    default=MODEL_NAME,
    type=str,
    help=f"The model to be used in the application. Defaults to {MODEL_NAME}",
)
parser.add_argument(
    "--max-new-tokens",
    default=MAX_NEW_TOKENS,
    type=int,
    help=f"Number of tokens to be used in each generation round. Defaults to {MAX_NEW_TOKENS}",
)
parser.add_argument(
    "--seed",
    type=int,
    help="Seed for text generation. Optional, use for reproducibility.",
)


def main():
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, device_map="auto", torch_dtype=torch.bfloat16)

    if args.seed is not None:
        set_seed(args.seed)

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
