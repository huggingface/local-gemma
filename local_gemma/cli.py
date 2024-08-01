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
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "1"

import argparse
import sys

import torch
from transformers import AutoTokenizer, TextStreamer, set_seed
from transformers.cache_utils import HybridCache
from transformers.utils import logging, is_flash_attn_2_available
from accelerate.utils import is_torch_version

from huggingface_hub import get_token, login
from local_gemma import LocalGemma2ForCausalLM
from .utils.benchmark import benchmark
from .utils.config import (
    DTYPE_MODIFIER, infer_device, infer_dtype, get_prompt, get_generation_kwargs, infer_memory_requirements
)

torch.set_float32_matmul_precision("high")


EXIT_COMMANDS = {"!exit", "quit", "quit()", "!exit()", "!quit", "!quit()"}
NEW_SESSION_COMMANDS = {"!new session", "!new session()", "!new chat", "!new chat()", "!new", "!reset"}
MODEL_NAMES = {
    "2b": "google/gemma-2-2b-it",
    "9b": "google/gemma-2-9b-it",
    "27b": "google/gemma-2-27b-it",
}

parser = argparse.ArgumentParser(description="Local Gemma")

# Prompt argument
parser.add_argument(
    "prompt",
    type=str,
    nargs="*",
    help=(
        "Prompt to the model. For an interactive session, leave this field empty."
    ),
)
# Other control arguments
parser.add_argument(
    "--model",
    type=str,
    default="9b",
    help=(
        "Size of Gemma 2 instruct model to be used in the application ('2b', '9b' or '27b') or, alternatively, a "
        "Hugging Face repo. Defaults to '9b'."
    ),
)
parser.add_argument(
    "--token",
    type=str,
    help="Authentication token for the model. Required to download the model into a local cache.",
)
parser.add_argument(
    "--preset",
    type=str,
    choices=["auto", "exact", "speed", "memory", "memory_extreme"],
    default="auto",
    help=(
        "Sets the optimization strategy for the local model deployment. Defaults to 'auto', which selects the best "
        "strategy for your device. 'exact' maximises accuracy, 'speed' maximizes speed, 'memory' reduces "
        "memory requirements through quantization, and 'memory_extreme' minimises memory requirements."
    ),
)
parser.add_argument(
    "--mode",
    type=str,
    choices=["chat", "factual", "creative"],
    default="chat",
    help=(
        "Sets the mode of operation of the model. 'chat' is optimized for general conversation, 'factual' is designed "
        "to minimize hallucinations, and 'creative' is optimized for creative writing. Note that 'factual' and "
        "'creative' prepend text to your prompt."
    ),
)
parser.add_argument(
    "--max_new_tokens",
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
    help="The dtype in which computations are performed. Defaults to the dtype set by --preset",
)
parser.add_argument(
    "--silent",
    action="store_true",
    help="Does NOT print any output except for the model outputs.",
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


def print_help(is_instruction_tuned: bool = True):
    if is_instruction_tuned:
        print("\nYou can now interact with the model through a conversation. A few tips:")
        print("- Initialize the program with '--silent' to hide all non-model messages")
        print("- Input '!exit' to leave the program")
        print("- Input '!new session' to reset the conversation")
        print("- Input '!help' to print this message again")
    else:
        print("\nYou can now pass a prompt to the base model to generate a single response.")
        print("Tip: for multi-turn conversation, use an instruction tuned model, such as `google/gemma-2-9b-it`.")
    print("")


def main():
    args = parser.parse_args()

    stdout_received = not sys.stdin.isatty()
    if stdout_received:
        input_data = sys.stdin.read()
        args.prompt = args.prompt + ["\n"] + [input_data]

    device = infer_device(args.device)
    dtype = infer_dtype(device, args.dtype)
    generation_kwargs = get_generation_kwargs(args.mode)
    base_prompt = get_prompt(args.mode)
    has_starting_prompt = len(args.prompt) > 0
    model_name = MODEL_NAMES.get(args.model) or args.model
    if args.token is None:
        if get_token() is None:
            print("Using the gated Gemma model requires you to:")
            print("1. Create an account on the Hugging Face Hub: https://huggingface.co/join")
            print("2. Accept the Gemma-2 model terms of use: https://huggingface.co/google/gemma-2-9b")
            print("3. Create an access token and paste it below: https://huggingface.co/settings/tokens")
            login()

    if args.preset == "auto":
        args.preset, spare_memory = infer_memory_requirements(
            model_name, device, trust_remote_code=False, token=args.token
        )

    # Triggers assisted generation on CUDA or MPS devices, assuming the default 9b or 27b models are used. Assisted
    # generation is not beneficial on most CPU settings. Can't be used with the speed preset (more precisely, with
    # `torch.compile`).
    if (
        args.model in ('9b', '27b')
        and ("cuda" in device or device.isdigit() or "mps" in device)
        and args.preset != "speed"
    ):
        assistant_model_name = MODEL_NAMES["2b"]
        if spare_memory / 1e9 > 5:
            assistant_preset = "exact"
        else:
            assistant_preset = "memory"
    else:
        assistant_model_name = None

    if not args.silent:
        print("\nLoading model with the following characteristics:")
        print("- Model name:", model_name)
        print(f"- Assistant model name: {assistant_model_name if assistant_model_name is None else assistant_model_name + f' (loaded with `{assistant_preset}` preset)'}")
        print("- Device:", device)
        print("- Default data type:", str(dtype))
        print("- Optimization preset:", args.preset)
        print("- Generation arguments:", str(generation_kwargs))
        print("- Base prompt:", repr(base_prompt) if len(base_prompt) > 0 else "None")
        print("")
    else:
        logging.disable_progress_bar()

    tokenizer = AutoTokenizer.from_pretrained(model_name, token=args.token)
    is_instruction_tuned = tokenizer.chat_template is not None

    if args.preset == "speed" and device == "cuda" and (has_starting_prompt or not is_instruction_tuned):
        # for single-turn responses, disable torch compile and enable fa2
        # this way, we skip the lengthy compilation step and return the generation to the user as quickly as possible
        torch_compile = False
        attn_implementation = "flash_attention_2" if is_flash_attn_2_available() else None
    else:
        # leave to the preset to decide these settings
        torch_compile = attn_implementation = None

    model = LocalGemma2ForCausalLM.from_pretrained(
        model_name,
        preset=args.preset,
        torch_compile=torch_compile,
        token=args.token,
        torch_dtype=dtype,
        device=device,
        attn_implementation=attn_implementation,
    )
    # TODO(joao): this if shouldn't be needed, fix in transformers
    model._supports_cache_class = True

    if assistant_model_name is not None:
        assistant_model = LocalGemma2ForCausalLM.from_pretrained(
            assistant_model_name, preset=assistant_preset, token=args.token, torch_dtype=dtype, device=device)
    else:
        assistant_model = None

    if args.benchmark:
        benchmark(model=model, assistant_model=assistant_model, tokenizer=tokenizer)
    else:
        if args.seed is not None:
            set_seed(args.seed)

        if device == "mps" and args.max_new_tokens is None:
            print(
                "Setting max new tokens to 1024 for faster mps generation. To bypass this limit, set "
                "`--max_new_tokens=2048`."
            )
            args.max_new_tokens = 1024

        # Note: as of transformers 4.44, assisted generation does NOT work with any cache except dynamic cache
        if args.max_new_tokens is None and assistant_model is None:
            cache = HybridCache(
                model.config,
                max_batch_size=1,
                max_cache_len=model.config.max_position_embeddings,
                device=model.device,
                dtype=model.dtype,
            )
            model.generation_config.cache_implementation = None
        else:
            # when generating using max_new_tokens, update the cache on each generation step to limit memory
            cache = None

        if hasattr(model.forward, "_torchdynamo_orig_callable"):
            print(
                "Compiling the model forward pass. This may take a few minutes, particularly the first time it is "
                "run..."
            )
            if not is_torch_version(">=", "2.4"):
                print(
                    "Install torch>=2.4.0 to cache the FX graphs across runs: https://pytorch.org/get-started/locally/"
                )
            chat_history = [{"role": "user", "content": "The theory of relativity states"}, ]
            # Two warm-up runs: First run warms up model (triton autotuning etc). Second run records the graph and plays it. The third run is the fast path...
            for _ in range(2):
                dummy_inputs = tokenizer.apply_chat_template(chat_history, tokenize=False, add_generation_prompt=True)
                model_inputs = tokenizer(dummy_inputs, return_tensors="pt").to(model.device)
                # prefill + generation
                model_tokens = model.generate(**model_inputs, past_key_values=cache, max_new_tokens=16)
                model_output_text = tokenizer.decode(model_tokens[0, model_inputs.input_ids.shape[1]:], skip_special_tokens=True)
                chat_history += [{"role": "assistant", "content": model_output_text},  {"role": "user", "content": "Please repeat the above!"},]
                cache.reset()

        if not args.silent and not has_starting_prompt:
            print_help(is_instruction_tuned=is_instruction_tuned)

        streamer = TextStreamer(tokenizer, skip_prompt=True, **{"skip_special_tokens": True})
        chat_history = []
        while True:
            # Get input to the model
            if has_starting_prompt:
                user_input = " ".join(args.prompt)
            else:
                user_input = input(">>> ")

            # Handle special commands
            if user_input in EXIT_COMMANDS:
                break
            elif user_input in NEW_SESSION_COMMANDS:
                chat_history = []
                if hasattr(cache, "reset"):
                    cache.reset()
                else:
                    cache = None
            elif user_input == "!help":
                print_help()

            # Generate text
            else:
                # Inject the base prompt if the chat history is empty
                if len(chat_history) == 0:
                    user_input = base_prompt + user_input

                chat_history += [{"role": "user", "content": user_input},]
                if is_instruction_tuned:
                    user_input = tokenizer.apply_chat_template(chat_history, tokenize=False, add_generation_prompt=True)

                model_inputs = tokenizer(
                    user_input,
                    return_tensors="pt",
                    return_attention_mask=device == "mps",
                )
                input_ids = model_inputs.input_ids

                model_inputs = model_inputs.to(device)
                generation_kwargs.update(
                    {
                        "streamer": streamer,
                        "assistant_model": assistant_model,
                        "past_key_values": cache,
                    }
                )
                if args.max_new_tokens is not None:
                    generation_kwargs["max_new_tokens"] = args.max_new_tokens
                    input_ids_len = input_ids.shape[-1]
                    max_cache_len = args.max_new_tokens + input_ids_len
                    if cache is not None and cache.max_cache_len < max_cache_len:
                        # reset the cache
                        generation_kwargs.pop("past_key_values")
                        generation_kwargs["cache_implementation"] = "hybrid"
                else:
                    generation_kwargs["max_length"] = model.config.max_position_embeddings

                model_tokens = model.generate(**model_inputs, **generation_kwargs)
                model_tokens = model_tokens[0, input_ids.shape[1]:]
                model_output_text = tokenizer.decode(model_tokens, skip_special_tokens=True)
                chat_history += [{"role": "assistant", "content": model_output_text},]

                if is_instruction_tuned:
                    # Sanity check: EOS was removed, ends in "<end_of_turn>\n"
                    tokenized_chat = tokenizer.apply_chat_template(
                        chat_history, tokenize=True, add_generation_prompt=False, return_tensors="pt"
                    ).tolist()[0]
                    assert tokenized_chat[0] == 2
                    assert tokenized_chat[-1] == 108
                    assert tokenized_chat[-2] == 107

            if has_starting_prompt or not is_instruction_tuned:
                break

if __name__ == '__main__':
    main()
