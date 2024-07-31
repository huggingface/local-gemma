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

from time import time
from tqdm import tqdm


PROMPT_LENGTH = [64, 2048]
MAX_NEW_TOKENS = [64, 2048]
NUM_RUNS = 5
WARMUP_RUNS = 3


def benchmark(model, assistant_model, tokenizer):
    """
    Benchmarkes the throughput of the model. Does some warmup before measuring, to remove compilation time (if
    applicable).
    """
    max_prompt_length = max(PROMPT_LENGTH)
    model_inputs = tokenizer(
        ["foo bar " * 4000], return_tensors="pt", truncation=True, max_length=max_prompt_length
    )

    # sanity check
    if model_inputs.input_ids.shape[1] != max_prompt_length:
        raise ValueError(
            f"Benchmark error: Model input length is {model_inputs.input_ids.shape[1]}, but expected to be "
            f"{max_prompt_length}."
        )

    # benchmark
    results = {}
    for prompt_length in PROMPT_LENGTH:
        for max_new_tokens in MAX_NEW_TOKENS:
            print(f"\nBenchmarking with prompt_length={prompt_length} and max_new_tokens={max_new_tokens}.")
            run_name = f"prompt_length={prompt_length}, max_new_tokens={max_new_tokens}"
            generate_kwargs = {
                "do_sample": False,
                "max_new_tokens": max_new_tokens,
                "min_new_tokens": max_new_tokens,
                "assistant_model": assistant_model,
            }

            input_ids = model_inputs.input_ids[:, :prompt_length].to(model.device)
            for _ in tqdm(range(WARMUP_RUNS), desc="Warming up"):
                model.generate(input_ids, **generate_kwargs)

            tokens_per_second = []
            for _ in tqdm(range(NUM_RUNS), desc="Benchmarking"):
                start = time()
                gen_out = model.generate(input_ids, **generate_kwargs)
                end = time()
                if gen_out.shape[1] != prompt_length + max_new_tokens:
                    raise ValueError(
                        f"Benchmark error: Generated output length is {gen_out.shape[1]}, but expected to be "
                        f"{prompt_length + max_new_tokens}."
                    )
                tokens_per_second.append((max_new_tokens) / (end - start))
            results[run_name] = sum(tokens_per_second)/len(tokens_per_second)
            print(f"{run_name:40s}: {results[run_name]:2f} tokens per second.\n")

    # print results
    print("\n\nResults:")
    for run_name, throughput in results.items():
        print(f"{run_name:40s}: {throughput:2f} tokens per second.")
