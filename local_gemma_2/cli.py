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

from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer, set_seed
from transformers.utils import logging

from .utils.benchmark import benchmark
from .utils.config import infer_device, infer_dtype, infer_attention_type, get_prompt, get_generation_kwargs


MODEL_NAMES = {
    "9b": "google/gemma-2-9b-it",
    "27b": "google/gemma-2-27b-it",
}


parser = argparse.ArgumentParser(description="Local Gemma 2")

# Prompt argument
parser.add_argument(
    "prompt",
    type=str,
    nargs="*",
    help=(
        "Prompt to the model. For an interactive session, leave this field empty. Using this field will activate "
        "'--silent'"
    ),
)
# Other control arguments
parser.add_argument(
    "--model",
    type=str,
    default="9b",
    help=(
        "Size of Gemma 2 model to be used in the application ('9b' or '27'b) or, alternatively, a Hugging Face repo. "
        "Defaults to '9b'."
    ),
)
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
    choices=["chat", "factual", "creative"],
    default="chat",
    help=(
        "Sets the mode of operation of the model. 'chat' is optimized for general conversation, 'factual' is designed "
        "to minimize hallucinations, and 'creative' is optimized for creative writing. Note that 'factual' and "
        "'creative' prepend text to your prompt."
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


def print_help():
    print("\nYou can now interact with the model. A few tips:")
    print("- Initialize the program with '--silent' to hide all non-model messages")
    print("- Input '!exit' to leave the program")
    print("- Input '!new session' to reset the conversation")
    print("- Input '!help' to print this message again")
    print("")


def main():
    args = parser.parse_args()
    if args.prompt:
        args.silent = True

    device = infer_device(args.device)
    dtype = infer_dtype(args.dtype)
    attention_type = infer_attention_type(device)
    generation_kwargs = get_generation_kwargs(args.mode)
    base_prompt = get_prompt(args.mode)
    model_name =  MODEL_NAMES.get(args.model) or args.model

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
        logging.disable_progress_bar()  # TODO(joao): this is not working, should suppress "Loading checkpoint shards"
        print("\nLoading model with the following characteristics:")
        print("- Model name:", model_name)
        # print("- Assistant model name:", assistant_model_name)
        print("- Device:", device)
        print("- Data type:", str(dtype))
        print("- Attention type:", attention_type)
        print("- Generation arguments:", str(generation_kwargs))
        print("- Base prompt:", repr(base_prompt) if len(base_prompt) > 0 else "None")
        print("")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, attn_implementation=attention_type, token=args.auth_token
    ).to(device)
    # if assistant_model_name is not None:
    #     assistant_model = AutoModelForCausalLM.from_pretrained(
    #         assistant_model_name, torch_dtype=dtype, attn_implementation=attention_type, token=args.auth_token
    #     ).to(device)
    # else:
    assistant_model = None

    if args.benchmark:
        benchmark(model, tokenizer)
    else:
        if args.seed is not None:
            set_seed(args.seed)

        if not args.silent:
            print_help()

        has_starting_prompt = len(args.prompt) > 0
        streamer = TextStreamer(tokenizer, skip_prompt=True, **{"skip_special_tokens": True})
        cache = None
        chat_history = []
        while True:
            # Get input to the model
            if has_starting_prompt:
                user_input = " ".join(args.prompt)
            else:
                # Try/except to allow piping on bash, like `echo "1+1=" | local-gemma-2`
                try:
                    user_input = input(">>> ")
                except EOFError:
                    break

            # Handle special commands
            if user_input == "!exit":
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
                tokenized_chat = tokenizer.apply_chat_template(
                    chat_history, tokenize=True, add_generation_prompt=True, return_tensors="pt"
                )
                tokenized_chat = tokenized_chat.to(device)
                generation_kwargs.update(
                    {
                        "streamer": streamer,
                        "assistant_model": assistant_model,
                        "return_dict_in_generate": True,
                        "past_key_values": cache,
                    }
                )
                if args.max_new_tokens is not None:
                    generation_kwargs["max_new_tokens"] = args.max_new_tokens
                else:
                    generation_kwargs["max_length"] = model.config.max_position_embeddings

                gen_out = model.generate(input_ids=tokenized_chat, **generation_kwargs)

                # Store the cache for the next generation round; Pull the model output into the chat history.
                cache = gen_out.past_key_values
                model_tokens = gen_out.sequences[0, tokenized_chat.shape[1]:]
                model_output_text = tokenizer.decode(model_tokens, skip_special_tokens=True)
                chat_history += [{"role": "assistant", "content": model_output_text},]

                # Sanity check: EOS was removed, ends in "<end_of_turn>\n"
                tokenized_chat = tokenizer.apply_chat_template(
                    chat_history, tokenize=True, add_generation_prompt=False, return_tensors="pt"
                ).tolist()[0]
                assert tokenized_chat[0] == 2
                assert tokenized_chat[-1] == 108
                assert tokenized_chat[-2] == 107

            if has_starting_prompt:
                break

if __name__ == '__main__':
    main()
