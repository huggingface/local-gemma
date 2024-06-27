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
from typing import Optional, Union, Dict
import logging

import torch
from transformers import QuantoConfig, is_bitsandbytes_available, BitsAndBytesConfig
from transformers.utils import is_quanto_available, is_torch_sdpa_available, is_accelerate_available
from transformers.models.gemma import GemmaForCausalLM, GemmaConfig
from .utils.config import infer_device, infer_dtype


logger = logging.getLogger(__name__)

EXACT = {
    "attn_implementation": "eager",
}

SPEED = {
    "attn_implementation": "sdpa",
}

MEMORY = {
    "attn_implementation": "sdpa",
    "quantization_config": {
        "weights": "int4"
    }
}

MEMORY_EXTREME = {
    "attn_implementation": "sdpa",
    "quantization_config": {
        "weights": "int2"
    }
}


PRESET_MAPPING = {
    "exact": EXACT,
    "speed": SPEED,
    "memory": MEMORY,
    "memory_extreme": MEMORY,
}

class LocalGemma2ForCausalLM(GemmaForCausalLM):
    @staticmethod
    # TODO(SG): potentially bypass these checks by pinning requirements
    def get_preset_kwargs(preset: str, device: str) -> Dict:
        preset_kwargs = PRESET_MAPPING.get(preset)
        if preset_kwargs is None:
            raise ValueError(f"Got invalid `preset` {preset}. Ensure `preset` is one of: {list(PRESET_MAPPING.keys())}")
        if preset == "speed" and not is_torch_sdpa_available():
            raise ImportError(
                "The 'speed' preset requires PyTorch v2.1.1 or later. Please install torch>=2.1.1 through the "
                "official instructions: https://pytorch.org/"
            )
        if preset in ["memory", "memory_extreme"]:
            if not is_torch_sdpa_available():
                logger.warning(
                    "Detected PyTorch version <2.1.1. For faster inference through SDPA attention, install PyTorch "
                    "v2.1.1 or later through the official instructions: https://pytorch.org/"
                )
                preset_kwargs["attn_implementation"] = "eager"
            if device == "cuda" and not is_bitsandbytes_available():
                raise ImportError(
                    f"The {preset} preset on CUDA requires the `bitsandbytes` package. Please install bitsandbytes through: "
                    "`pip install --upgrade bitsandbytes`."
                )
            elif device != "cuda" and not is_quanto_available():
                raise ImportError(
                    f"The {preset} preset on {device} requires the `quanto` package. Please install quanto through: "
                    "`pip install --upgrade quanto`."
                )
        if preset == "memory_extreme":
            if not is_accelerate_available():
                raise ImportError(
                    f"The `memory_extreme` preset requires the `accelerate` package. Please install accelerate through: "
                    "`pip install --upgrade accelerate`."
                )
            if not is_quanto_available():
                raise ImportError(
                    f"The `memory_extreme` preset on {device} requires the `quanto` package. Please install quanto through: "
                    "`pip install --upgrade quanto`."
                )
            if device == "cuda":
                preset_kwargs["device_map"] = "auto"
        return preset_kwargs

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        preset: Optional[str] = "exact",
        *model_args,
        config: Optional[Union[GemmaConfig, str, os.PathLike]] = None,
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        ignore_mismatched_sizes: bool = False,
        force_download: bool = False,
        local_files_only: bool = False,
        token: Optional[Union[str, bool]] = None,
        revision: str = "main",
        use_safetensors: bool = None,
        **kwargs,
    ) -> GemmaForCausalLM:
        device = infer_device()
        preset_kwargs = cls.get_preset_kwargs(preset, device)

        torch_dtype = kwargs.pop("torch_dtype", None)
        preset_kwargs["torch_dtype"] = infer_dtype(torch_dtype)

        quantization_config = kwargs.pop("quantization_config", None)
        if quantization_config is not None:
            preset_kwargs["quantization_config"] = quantization_config
        elif preset_kwargs.get("quantization_config"):
            if device == "cuda" and preset == "memory":
                preset_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=preset_kwargs["torch_dtype"])
            else:
                preset_kwargs["quantization_config"] = QuantoConfig(weights=preset_kwargs["quantization_config"]["weights"])

        if kwargs is not None:
            for key in kwargs:
                if key in preset_kwargs:
                    preset_kwargs[key] = kwargs[key]

        model = super().from_pretrained(
            pretrained_model_name_or_path,
            *model_args,
            config=config,
            cache_dir=cache_dir,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
            force_download=force_download,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
            use_safetensors=use_safetensors,
            **preset_kwargs,
            **kwargs,
        )

        # TODO(SG): decide on automatic device placement
        model = model.to(device)

        if preset != "memory_extreme" and device == "cuda":
            model.generation_config.cache_implementation = "static"
            model.forward = torch.compile(model.forward, mode="reduce-overhead", fullgraph=True)
        elif preset == "memory_extreme":
            model.generation_config.cache_implementation = "quantized"

        return model