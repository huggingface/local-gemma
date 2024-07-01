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
import sys

import torch
from transformers import AutoTokenizer, TextStreamer, set_seed
from transformers.utils import logging

from local_gemma import LocalGemma2ForCausalLM
from .utils.benchmark import benchmark
from .utils.config import infer_device, infer_dtype, get_prompt, get_generation_kwargs, infer_memory_requirements


MODEL_NAMES = {
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
        "Size of Gemma 2 instruct model to be used in the application ('9b' or '27'b) or, alternatively, a Hugging "
        "Face repo. Defaults to '9b'."
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
        "strategy for your device. 'exact' maximises accuracy, 'speed' maximises generation speed, 'memory' reduces "
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
    model_name = MODEL_NAMES.get(args.model) or args.model

    if args.preset == "auto":
        args.preset = infer_memory_requirements(model_name, device, trust_remote_code=False, token=args.token)

    # TODO(joao) : assisted generation
    # # Triggers assisted generation on CUDA or MPS devices, assuming the default model is used. Assisted generation is
    # # not beneficial on most CPU settings.
    # if  args.model_name is None and ("cuda" in device or device.isdigit() or "mps" in device):
    #     assistant_model_name = (
    #         ASSISTANT_MODEL_NAME if args.optimization == "quality" else QUANTIZED_ASSISTANT_MODEL_NAME
    #     )
    # else:
    #     assistant_model_name = None

    if not args.silent:
        print("\nLoading model with the following characteristics:")
        print("- Model name:", model_name)
        # print("- Assistant model name:", assistant_model_name)
        print("- Device:", device)
        print("- Data type:", str(dtype))
        print("- Optimization preset:", args.preset)
        print("- Generation arguments:", str(generation_kwargs))
        print("- Base prompt:", repr(base_prompt) if len(base_prompt) > 0 else "None")
        print("")
    else:
        logging.disable_progress_bar()

    tokenizer = AutoTokenizer.from_pretrained(model_name, token=args.token)
    model = LocalGemma2ForCausalLM.from_pretrained(
        model_name, preset=args.preset, token=args.token, torch_dtype=dtype, device=device
    )
    # TODO(joao): this if shouldn't be needed, fix in transformers
    model._supports_cache_class = True

    # if assistant_model_name is not None:
    #     assistant_model = LocalGemma2ForCausalLM.from_pretrained(
    #         assistant_model_name, preset=args.preset, token=args.token, torch_dtype=dtype, device=device)
    # else:
        # assistant_model = None
    assistant_model = None

    if args.benchmark:
        benchmark(model, tokenizer)
    else:
        if args.seed is not None:
            set_seed(args.seed)

        has_starting_prompt = len(args.prompt) > 0
        is_instruction_tuned = tokenizer.chat_template is not None

        if device == "mps" and args.preset == "auto" and args.max_new_tokens is None:
            print("Setting max new tokens to 1024 for faster mps generation. To bypass this limit, set `--max_new_tokens=2048`.")
            args.max_new_tokens = 1024

        if not args.silent and not has_starting_prompt:
            print_help(is_instruction_tuned=is_instruction_tuned)

        streamer = TextStreamer(tokenizer, skip_prompt=True, **{"skip_special_tokens": True})
        cache = None
        chat_history = []
        while True:
            # Get input to the model
            if has_starting_prompt:
                user_input = " ".join(args.prompt)
            else:
                user_input = input(">>> ")

            # Handle special commands
            if user_input in ["!exit", "quit", "quit()"]:
                break
            elif user_input == "!new session":
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
                    tokenized_chat = tokenizer.apply_chat_template(
                        chat_history, tokenize=True, add_generation_prompt=True, return_tensors="pt"
                    )
                else:
                    tokenized_chat = tokenizer(user_input, return_tensors="pt").input_ids

                tokenized_chat = tokenized_chat.to(device)
                generation_kwargs.update(
                    {
                        "streamer": streamer,
                        "assistant_model": assistant_model,
                        "return_dict_in_generate": True,
                        "past_key_values": cache,
                    }
                )
                # TODO(joao): this if shouldn't be needed, fix in transformers
                if cache is not None:
                    generation_kwargs["cache_implementation"] = None

                if args.max_new_tokens is not None:
                    generation_kwargs["max_new_tokens"] = args.max_new_tokens
                    input_ids_len = tokenized_chat.shape[-1]
                    max_cache_len = args.max_new_tokens + input_ids_len
                    if cache is not None and cache.max_cache_len < max_cache_len:
                        # reset the cache
                        generation_kwargs.pop("past_key_values")
                        generation_kwargs["cache_implementation"] = "hybrid"
                else:
                    generation_kwargs["max_length"] = model.config.max_position_embeddings

                if device == "mps":
                    generation_kwargs["attention_mask"] = torch.ones_like(tokenized_chat)

                gen_out = model.generate(input_ids=tokenized_chat, **generation_kwargs)

                # Store the cache for the next generation round; Pull the model output into the chat history.
                cache = gen_out.past_key_values
                model_tokens = gen_out.sequences[0, tokenized_chat.shape[1]:]
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
