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
from tqdm import tqdm

import torch
from transformers import QuantoConfig, is_bitsandbytes_available, BitsAndBytesConfig
from transformers.utils import is_quanto_available, is_accelerate_available
from transformers.models.gemma2.configuration_gemma2 import Gemma2Config
from transformers.models.gemma2.modeling_gemma2 import Gemma2ForCausalLM, GEMMA2_ATTENTION_CLASSES
import transformers.models.gemma2.modeling_gemma2
from .attention import Gemma2FusedAttention
from .utils.config import infer_device, infer_dtype, infer_memory_requirements


logger = logging.getLogger(__name__)

EXACT = {
    "attn_implementation": "eager",
}

SPEED = {
    "attn_implementation": "eager",
    "torch_compile": True,
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
            preset, _ = infer_memory_requirements(
                pretrained_model_name_or_path, device, trust_remote_code=trust_remote_code, token=token
            )
            logger.info(f"Detected device {device} and defaulting to {preset} preset.")

        preset_kwargs = PRESET_MAPPING[preset]

        if preset == "speed" and device != "cuda":
            # disable torch compile on non-cuda devices since it's not compatible
            preset_kwargs["torch_compile"] = False

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
        torch_compile: Optional[bool] = None,
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

        preset_torch_compile = preset_kwargs.pop("torch_compile", False)
        torch_compile = torch_compile if torch_compile is not None else preset_torch_compile

        quantization_config = kwargs.pop("quantization_config", None)
        if quantization_config is not None:
            preset_kwargs["quantization_config"] = quantization_config
        elif preset_kwargs.get("quantization_config"):
            if device == "cuda":
                preset_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    llm_int8_enable_fp32_cpu_offload=True,
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

        if torch_compile and device != "cuda":
            raise ValueError(
                "Torch compile is only compatible with cuda devices. Set `torch_compile=False` in `.from_pretrained`"
                f"for device {device}."
            )
        elif torch_compile:
            model = fuse_attention_weights(model, device, torch_dtype)
            model.forward = torch.compile(model.forward, mode="reduce-overhead", fullgraph=True)

        return model

def fuse_attention_weights(model: LocalGemma2ForCausalLM, device, torch_dtype) -> LocalGemma2ForCausalLM:
    for idx, layer in tqdm(enumerate(model.model.layers), desc="Fusing attention weights", total=model.config.num_hidden_layers):
        state_dict = layer.self_attn.state_dict()
        del layer.self_attn
        layer.self_attn = Gemma2FusedAttention(model.config, layer_idx=idx)
        # convert un-fused to fused through the pre-register hook
        layer.self_attn.load_state_dict(state_dict)
        layer.self_attn.to(device, dtype=torch_dtype)
    return model
