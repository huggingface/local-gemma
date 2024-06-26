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
from transformers.utils import is_bitsandbytes_available, is_torch_sdpa_available, is_accelerate_available
from transformers.models.gemma import GemmaForCausalLM, GemmaConfig
from .utils.config import infer_device, infer_dtype, infer_attention_type

logger = logging.getLogger(__name__)

EXACT = {
    "attn_implementation": "eager",
}

SPEED = {
    "attn_implementation": "sdpa",
}

MEMORY = {
    "attn_implementation": "sdpa",
    "load_in_4_bit": True,
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
    def check_preset(preset: str, preset_kwargs: Dict) -> Dict:
        if preset_kwargs is None:
            raise ValueError(f"Got invalid `preset` {preset}. Ensure `preset` is one of: {list(PRESET_MAPPING.keys())}")
        if preset == "speed" and not is_torch_sdpa_available():
            raise ImportError(
                "The 'speed' preset requires PyTorch v2.1.1 or later. Please install torch>=2.1.1 through the "
                "official instructions: https://pytorch.org/"
            )
        if preset == "memory":
            if not is_torch_sdpa_available():
                logger.warning(
                    "Detected PyTorch version <2.1.1. For faster inference through SDPA attention, install PyTorch "
                    "v2.1.1 or later through the official instructions: https://pytorch.org/"
                )
                preset_kwargs["attn_implementation"] = "eager"
            # TODO(SG): determine the best quantisation scheme (bnb vs quanto vs awq vs etc.), e.g. on a per-device basis?
            if not is_bitsandbytes_available():
                raise ImportError(
                    "The 'memory' preset requires the `bitsandbytes` package. Please install bitsandbytes through: "
                    "`pip install --upgrade bitsandbytes`."
                )
        return preset_kwargs

    def enable_cpu_offload(self, gpu_id: Optional[int] = 0):
        r"""
        Offloads all sub-models to CPU using accelerate, reducing memory usage with a low impact on performance. This
        method moves one whole sub-model at a time to the GPU when it is used, and the sub-model remains in GPU until
        the next sub-model runs.

        Args:
            gpu_id (`int`, *optional*, defaults to 0):
                GPU id on which the sub-models will be loaded and offloaded.
        """
        if is_accelerate_available():
            from accelerate import cpu_offload_with_hook
        else:
            raise ImportError("`enable_model_cpu_offload` requires `accelerate`.")

        device = torch.device(f"cuda:{gpu_id}")
        self.to("cpu")

        # clear cache, otherwise we don't see the memory savings (but they probably exist)
        torch.cuda.empty_cache()

        self.model.embed_tokens = cpu_offload_with_hook(self.model.embed_tokens, device)

        hook = None
        for layer in self.model.layers:
            _, hook = cpu_offload_with_hook(layer, device, prev_module_hook=hook)

        # TODO(SG): finish

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        *model_args,
        preset: Optional[str] = "exact",
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
        preset_kwargs: Dict = PRESET_MAPPING.get(preset)
        preset_kwargs = cls.check_preset(preset, preset_kwargs)

        torch_dtype = kwargs.pop("torch_dtype", None)
        preset_kwargs["torch_dtype"] = infer_dtype(torch_dtype)

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

        device = infer_device()
        model.to(device)

        if preset not in ["memory", "memory_extreme"] and device == "cuda":
            model.generation_config.cache_implementation = "static"
            model.forward = torch.compile(model.forward, mode="reduce-overhead", fullgraph=True)
        elif preset in ["memory", "memory_extreme"]:
            model.generation_config.cache_implementation = "quantized"

        if preset == "memory_extreme":
            # TODO(SG): register accelerate hooks for cpu offloading
            model = model

        return model