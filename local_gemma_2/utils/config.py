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


import torch

from typing import Optional

from transformers.utils import is_flash_attn_2_available


def infer_device(device: Optional[str] = None) -> str:
    """
    Infers basic devices available on the system. Prioritizes the most performant device.
    """
    if device is not None:
        return device
    # TODO: infer more dtypes
    elif torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# TODO(SG): ensure compatible dtypes with device
def infer_dtype(dtype: Optional[str] = None) -> torch.dtype:
    if dtype is None:
        return torch.float16
    # TODO: enable more dtypes
    elif dtype == "float32":
        return torch.float32
    elif dtype == "float16":
        return torch.float16
    elif dtype == "bfloat16":
        return torch.bfloat16

def infer_attention_type(device: str) -> str:
    if device == "cuda" and is_flash_attn_2_available():
        return "flash_attention_2"
    return "sdpa"
