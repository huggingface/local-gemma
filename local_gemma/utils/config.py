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
import sys
import psutil
import torch

from typing import Dict, Optional

from accelerate import init_empty_weights
from transformers import Gemma2ForCausalLM, Gemma2Config
from transformers.utils import is_torch_bf16_available_on_device
from accelerate.utils import calculate_maximum_sizes


DTYPE_MODIFIER = {"exact": 2, "speed": 2, "memory": 8, "memory_extreme": 8}
DTYPE_MAP = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


def infer_device(device: Optional[str] = None) -> str:
    """
    Infers basic devices available on the system. Prioritizes the most performant device.
    """
    if device is not None:
        return device
    elif torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def infer_dtype(device: str, dtype_str: Optional[str] = None) -> torch.dtype:
    if dtype_str is None:
        if is_torch_bf16_available_on_device(device):
            return torch.bfloat16
        else:
            return torch.float16
    dtype = DTYPE_MAP.get(dtype_str, None)
    if dtype is None:
        raise ValueError(f"Unknown dtype: {dtype_str}. Must be one of {DTYPE_MAP.keys()}")
    return dtype


def get_prompt(mode: str) -> str:
    if mode == "chat":
        return ""
    elif mode == "factual":
        return "Please reply to the following requests with short and factual answers.\n\n"
    elif mode == "creative":
        return (
            "Write a response that appropriately completes the request. Be descriptive, fluid, and follow the context "
            "provided.\n\n"
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")


def get_generation_kwargs(mode: str) -> Dict:
    generation_kwargs = {"do_sample": True}
    if mode == "chat":
        generation_kwargs["temperature"] = 0.7
    elif mode == "factual":
        generation_kwargs["temperature"] = 0.3
        generation_kwargs["repetition_penalty"] = 1.2
    elif mode == "creative":
        generation_kwargs["min_p"] = 0.08
        generation_kwargs["temperature"] = 1.5
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return generation_kwargs


def infer_memory_requirements(model_name, device=None, token=None, trust_remote_code=False) -> str:
    config = Gemma2Config.from_pretrained(model_name, token=token, trust_remote_code=trust_remote_code)

    with init_empty_weights():
        model = Gemma2ForCausalLM(config)

    total_size, _ = calculate_maximum_sizes(model)
    device = infer_device(device)

    if device == "cuda":
        total_memory = torch.cuda.get_device_properties(device).total_memory
    else:
        total_memory = psutil.virtual_memory().total

    for preset in DTYPE_MODIFIER.keys():
        dtype_total_size = total_size / DTYPE_MODIFIER[preset]
        inference_requirements = 1.15 * dtype_total_size  # 1.15 allows A10G to run the `exact` preset on the 9b model
        spare_memory = total_memory - inference_requirements

        if inference_requirements < total_memory:
            return preset, spare_memory

    # if the model does not fit fully in the device, return the last preset ('memory_extreme') which will automatically
    # enable CPU offloading so that we can fit any device
    return "memory_extreme", 0
