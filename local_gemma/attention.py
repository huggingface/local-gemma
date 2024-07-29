from typing import Optional, Tuple
import logging

import torch
from torch import nn
from transformers import Cache
from transformers.models.gemma2.configuration_gemma2 import Gemma2Config
from transformers.models.gemma2.modeling_gemma2 import Gemma2RotaryEmbedding, apply_rotary_pos_emb, repeat_kv, GEMMA2_ATTENTION_CLASSES

logger = logging.getLogger(__name__)

class Gemma2FusedAttention(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper. Modified from the original implementation
    to include fused q/k/v projection.
    """

    def __init__(self, config: Gemma2Config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.scaling = config.query_pre_attn_scalar**-0.5

        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        # fused attention proj
        total_head_dim = (2 * self.num_key_value_heads + self.num_heads) * self.head_dim
        self.qkv_proj = nn.Linear(self.hidden_size, total_head_dim, bias=config.attention_bias)

        # conversion from un-fused to fused
        self._register_load_state_dict_pre_hook(self.load_hook)

        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)
        self.rotary_emb = Gemma2RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )
        self.sliding_window = config.sliding_window if not bool(layer_idx % 2) else None

    def load_hook(self, state_dict, prefix, *args):
        if prefix + "q_proj.weight" in state_dict:
            q_proj = state_dict.pop(prefix + "q_proj.weight")
            k_proj = state_dict.pop(prefix + "k_proj.weight")
            v_proj = state_dict.pop(prefix + "v_proj.weight")
            state_dict[prefix + "qkv_proj.weight"] = torch.cat([q_proj, k_proj, v_proj])

            if self.config.attention_bias:
                q_bias = state_dict.pop(prefix + "q_proj.bias")
                k_bias = state_dict.pop(prefix + "k_proj.bias")
                v_bias = state_dict.pop(prefix + "v_proj.bias")
                state_dict[prefix + "qkv_proj.bias"] = torch.cat([q_bias, k_bias, v_bias])

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_key_value_states = self.qkv_proj(hidden_states)
        query_size = self.num_heads * self.head_dim
        key_value_size = self.num_key_value_heads * self.head_dim
        query_states, key_states, value_states = query_key_value_states.split(
            [query_size, key_value_size, key_value_size], dim=-1
        )

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {
                "sin": sin,
                "cos": cos,
                "sliding_window": self.sliding_window,
                "cache_position": cache_position,
            }
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling

        if self.config.attn_logit_softcapping is not None:
            attn_weights = attn_weights / self.config.attn_logit_softcapping
            attn_weights = torch.tanh(attn_weights)
            attn_weights = attn_weights * self.config.attn_logit_softcapping

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.view(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value
