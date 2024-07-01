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
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from typing import Optional, Union, Dict
import logging

import torch
from transformers import QuantoConfig, is_bitsandbytes_available, BitsAndBytesConfig
from transformers.utils import is_quanto_available, is_torch_sdpa_available, is_accelerate_available
from transformers.models.gemma2 import Gemma2ForCausalLM, Gemma2Config
from .utils.config import infer_device, infer_dtype, infer_memory_requirements


logger = logging.getLogger(__name__)

EXACT = {
    "attn_implementation": "eager",
}

MEMORY = {
    "attn_implementation": "eager",
    "quantization_config": {
        "weights": "int4"
    }
}

MEMORY_EXTREME = {
    "attn_implementation": "eager",
    "device_map": "auto",
    "quantization_config": {
        "weights": "int4"
    }
}


PRESET_MAPPING = {
    "auto": None,
    "exact": EXACT,
    "memory": MEMORY,
    "memory_extreme": MEMORY_EXTREME,
}

class LocalGemma2ForCausalLM(Gemma2ForCausalLM):
    @staticmethod
    def get_preset_kwargs(pretrained_model_name_or_path: str, preset: str, device: str, trust_remote_code: bool = False, token: str = None) -> Dict:
        if preset not in PRESET_MAPPING:
            raise ValueError(f"Got invalid `preset` {preset}. Ensure `preset` is one of: {list(PRESET_MAPPING.keys())}")

        if preset == "auto":
            preset = infer_memory_requirements(pretrained_model_name_or_path, device, trust_remote_code=trust_remote_code, token=token)
            logger.info(f"Detected device {device} and defaulting to {preset} preset.")

        preset_kwargs = PRESET_MAPPING[preset]

        if preset in ["memory", "memory_extreme"]:
            if not is_torch_sdpa_available():
                logger.warning_once(
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

        return preset_kwargs

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        preset: Optional[str] = "auto",
        *model_args,
        config: Optional[Union[Gemma2Config, str, os.PathLike]] = None,
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        ignore_mismatched_sizes: bool = False,
        force_download: bool = False,
        local_files_only: bool = False,
        token: Optional[Union[str, bool]] = None,
        revision: str = "main",
        use_safetensors: bool = None,
        **kwargs,
    ) -> Gemma2ForCausalLM:
        device = infer_device(kwargs.pop("device", None))
        preset_kwargs = cls.get_preset_kwargs(
            pretrained_model_name_or_path,
            preset,
            device=device,
            trust_remote_code=kwargs.get("trust_remote_code"),
            token=kwargs.get("token"),
        )

        preset_kwargs["low_cpu_mem_usage"] = True
        torch_dtype = kwargs.pop("torch_dtype", None)
        if torch_dtype is None:
            torch_dtype = infer_dtype(device)
            if torch_dtype == torch.float16:
                extra_message = ' and weights' if preset not in ['memory', 'memory_extreme'] else ''
                logger.warning(
                    f"Defaulting to float16 precision for the computations{extra_message}. "
                    f"This can cause instabilities in generation for larger models, e.g. the 27b checkpoints."
                )
        preset_kwargs["torch_dtype"] = torch_dtype

        quantization_config = kwargs.pop("quantization_config", None)
        if quantization_config is not None:
            preset_kwargs["quantization_config"] = quantization_config
        elif preset_kwargs.get("quantization_config"):
            if device == "cuda":
                preset_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=preset_kwargs["torch_dtype"],
                )
            else:
                preset_kwargs["quantization_config"] = QuantoConfig(
                    weights=preset_kwargs["quantization_config"]["weights"]
                )

        # give preference to kwargs passed by the user
        kwargs_copy = kwargs.copy()
        if kwargs is not None:
            for key in kwargs_copy:
                if key in preset_kwargs:
                    preset_kwargs[key] = kwargs.pop(key)

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

        if device not in str(model.device) and preset_kwargs.get("device_map", None) is None:
            # for consistent behaviour with bitsandbytes, we move the model to the device always
            model.to(device, dtype=preset_kwargs["torch_dtype"])

        return model
