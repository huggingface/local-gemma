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
from transformers import QuantoConfig, is_bitsandbytes_available, BitsAndBytesConfig, is_torch_xla_available
from transformers.utils import is_quanto_available, is_accelerate_available
from transformers.models.gemma2.configuration_gemma2 import Gemma2Config
from transformers.models.gemma2.modeling_gemma2 import Gemma2ForCausalLM, Gemma2Model, GEMMA2_ATTENTION_CLASSES
import transformers.models.gemma2.modeling_gemma2
from .attention import Gemma2FusedAttention
from .utils.config import infer_device, infer_dtype, infer_memory_requirements


logger = logging.getLogger(__name__)

EXACT = {
    "attn_implementation": "eager",
}

SPEED = {
    "attn_implementation": "eager",  # TODO(SG): update to fused
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
    "speed": SPEED,
    "memory": MEMORY,
    "memory_extreme": MEMORY_EXTREME,
}

transformers.models.gemma2.modeling_gemma2.GEMMA2_ATTENTION_CLASSES = {
    **GEMMA2_ATTENTION_CLASSES,
    "fused": Gemma2FusedAttention,
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

        if preset == "speed" and device == "cuda":
            # TODO(SG): wrap compile here, or only in the CLI?
            model.forward = torch.compile(model.forward, mode="reduce-overhead", fullgraph=True)

        return model

    @classmethod
    def _autoset_attn_implementation(
        cls,
        config,
        use_flash_attention_2: bool = False,
        torch_dtype: Optional[torch.dtype] = None,
        device_map: Optional[Union[str, Dict[str, int]]] = None,
        check_device_map: bool = True,
    ):
        """
        Automatically checks and dispatches to a default attention implementation. In order of priority:
            1. An implementation specified in `config._attn_implementation` (due for example to the argument attn_implementation="sdpa" in from_pretrained).
            2. DEPRECATED: if use_flash_attention_2 is set to `True` and `flash_attn` is available, flash attention. (`LlamaFlashAttention` for example)
            3. SDPA implementation, if available and supported by the model type. (`LlamaSdpaAttention` for example)
            4. The default model's implementation otherwise (`LlamaAttention` for example) .
        """
        # Here we use config._attn_implementation_internal to check whether the attention implementation was explicitly set by the user.
        # The property `PretrainedConfig._attn_implementation` is never `None`, for backward compatibility (always fall back on "eager").
        # The `hasattr` here is used as some Transformers tests for some reason do not call PretrainedConfig __init__ (e.g. test_no_super_init_config_and_model)
        requested_attn_implementation = None
        if hasattr(config, "_attn_implementation_internal") and config._attn_implementation_internal is not None:
            if config._attn_implementation != "flash_attention_2" and use_flash_attention_2:
                raise ValueError(
                    f'Both attn_implementation="{config._attn_implementation}" and `use_flash_attention_2=True` were used when loading the model, which are not compatible.'
                    ' We recommend to just use `attn_implementation="flash_attention_2"` when loading the model.'
                )

            if config._attn_implementation not in ["eager", "fused", "sdpa", "flash_attention_2"]:
                message = f'Specified `attn_implementation="{config._attn_implementation}"` is not supported. The only possible arguments are `attn_implementation="eager"` (manual attention implementation), `attn_implementation="fused"` (fuse the query, key and value projections)'
                if cls._supports_flash_attn_2:
                    message += ', `"attn_implementation=flash_attention_2"` (implementation using flash attention 2)'
                if cls._supports_sdpa:
                    message += ', `"attn_implementation=sdpa"` (implementation using torch.nn.functional.scaled_dot_product_attention)'
                raise ValueError(message + ".")

            # If a config is passed with a preset attn_implementation, we skip the automatic dispatch and use the user-provided config, with hard checks that the requested attention implementation is available.
            requested_attn_implementation = config._attn_implementation_internal

        if use_flash_attention_2:
            logger.warning_once(
                'The model was loaded with use_flash_attention_2=True, which is deprecated and may be removed in a future release. Please use `attn_implementation="flash_attention_2"` instead.'
            )
            config._attn_implementation = "flash_attention_2"

        if config._attn_implementation == "flash_attention_2":
            cls._check_and_enable_flash_attn_2(
                config,
                torch_dtype=torch_dtype,
                device_map=device_map,
                hard_check_only=False,
                check_device_map=check_device_map,
            )
        elif requested_attn_implementation in [None, "sdpa"] and not is_torch_xla_available():
            # use_flash_attention_2 takes priority over SDPA, hence SDPA treated in this elif.
            config = cls._check_and_enable_sdpa(
                config,
                hard_check_only=False if requested_attn_implementation is None else True,
            )

            if torch.version.hip is not None and config._attn_implementation == "sdpa" and torch.cuda.device_count() > 1:
                logger.warning_once(
                    "Using the `SDPA` attention implementation on multi-gpu setup with ROCM may lead to performance issues due to the FA backend. Disabling it to use alternative backends."
                )
                torch.backends.cuda.enable_flash_sdp(False)

        return config

Gemma2Model._autoset_attn_implementation = LocalGemma2ForCausalLM._autoset_attn_implementation


