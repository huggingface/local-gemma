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

from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer, set_seed

from .utils.benchmark import benchmark
from .utils.config import infer_device, infer_dtype, infer_attention_type


FULL_MODEL_NAME = "google/gemma-1.1-7b-it"
ASSISTANT_MODEL_NAME = "google/gemma-1.1-2b-it"
QUANTIZED_MODEL_NAME = None
QUANTIZED_ASSISTANT_MODEL_NAME = None


parser = argparse.ArgumentParser(description="Local Gemma 2")

# Arguments that control text generation (sorted by importance)
parser.add_argument(
    "--auth-token",
    type=str,
    help="Authentication token for the model. Required to download the model into a local cache.",
)
parser.add_argument(
    "--optimization",
    type=str,
    choices=["quality", "speed", "memory"],
    default="quality",
    help=(
        "Sets the optimization strategy for the local model deployment. 'quality' loads the model without "
        "quantization and applies `torch.compile`. 'speed' optimizes throughput, using a quantized model. 'memory' "
        "applies further memory optimizations, like the quantized cache. Defaults to 'quality'."
    ),
)
parser.add_argument(
    "--mode",
    type=str,
    choices=["chat", "non-hallucinating", "creative"],
    default="chat",
    help=(
        "Sets the mode of operation of the model. 'chat' is optimized for general conversation, 'non-hallucinating' "
        "is designed to minimize hallucinations, and 'creative' is optimized for creative writing."
    ),
)
parser.add_argument(
    "--max-new-tokens",
    type=int,
    help=(
        "Maximum number of tokens to be used in each generation round. By default it relies on the model to emit an "
        "EOS token, and generates up to the pretrained length."
    ),
)
parser.add_argument(
    "--device",
    type=str,
    help="Forces a specific device to be used. By default uses cuda > mps > cpu, depending on availability.",
)
parser.add_argument(
    "--dtype",
    type=str,
    choices=["float32", "float16", "bfloat16"],
    default="float16",
    help="The dtype in which computations are performed. Defaults to float16."
)
parser.add_argument(
    "--model-name",
    type=str,
    help=f"Manually selects the model repo to be used in the application.",
)
# Debugging arguments
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
parser.add_argument(
    "--verbose",
    action="store_true",
    help="Prints information regarding the loaded model, selected arguments, and automatically handled exceptions.",
)

def main():
    args = parser.parse_args()

    device = infer_device(args.device)
    dtype = infer_dtype(args.dtype)
    attention_type = infer_attention_type(device)
    model_name = args.model_name or FULL_MODEL_NAME if args.optimization == "quality" else QUANTIZED_MODEL_NAME

    # Triggers assisted generation on CUDA or MPS devices, assuming the default model is used. Assisted generation is
    # not beneficial on most CPU settings.
    if  args.model_name is None and ("cuda" in device or device.isdigit() or "mps" in device):
        assistant_model_name = (
            ASSISTANT_MODEL_NAME if args.optimization == "quality" else QUANTIZED_ASSISTANT_MODEL_NAME
        )
    else:
        assistant_model_name = None

    if args.verbose:
        print("\nLoading model with the following characteristics:")
        print("- Model name:", model_name)
        print("- Assistant model name", assistant_model_name)
        print("- Device:", device)
        print("- Data type:", str(dtype))
        print("- Attention type:", attention_type)
        print("")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, attn_implementation=attention_type, token=args.auth_token
    ).to(device)
    if assistant_model_name is not None:
        assistant_model = AutoModelForCausalLM.from_pretrained(
            assistant_model_name, torch_dtype=dtype, attn_implementation=attention_type, token=args.auth_token
        ).to(device)
    else:
        assistant_model = None

    if args.benchmark:
        benchmark(model, tokenizer)
    else:
        if args.seed is not None:
            set_seed(args.seed)

        model_inputs = tokenizer("Hello world.", return_tensors="pt").to(model.device)
        streamer = TextStreamer(tokenizer, {"skip_special_tokens": True})
        generation_kwargs = {
            "do_sample": True,
            "streamer": streamer,
            "assistant_model": assistant_model
        }
        if args.max_new_tokens is not None:
            generation_kwargs["max_new_tokens"] = args.max_new_tokens
        else:
            generation_kwargs["max_length"] = model.config.max_position_embeddings

        gen_out = model.generate(**model_inputs, **generation_kwargs)


if __name__ == '__main__':
    main()
