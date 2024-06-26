# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer, set_seed

from .utils.benchmark import benchmark


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
parser.add_argument(
    "--benchmark",
    action="store_true",
    help="Runs a throughput benchmark on your device.",
)


def main():
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, device_map="auto", torch_dtype=torch.bfloat16)

    if args.benchmark:
        benchmark(model, tokenizer)
    else:
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
