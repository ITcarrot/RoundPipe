from typing_extensions import *
import warnings

import torch
import torch.nn as nn
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

from ..context import doing_recompute
from ..RoundPipe import RoundPipe
from .function import CompileForCausalLMLoss

class Qwen3ForCausalLMPrefix(nn.Module):
    def __init__(self, model: Qwen3ForCausalLM) -> None:
        super().__init__()
        self.embed_tokens = model.model.embed_tokens
        self.rotary_emb = model.model.rotary_emb
        self.config = model.model.config
        self.has_sliding_layers = model.model.has_sliding_layers

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Any] = None,
        inputs_embeds: Optional[torch.Tensor] = None, # pyright: ignore[reportRedeclaration]
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.Tensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Any,
    ):
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds: torch.Tensor = self.embed_tokens(input_ids)

        # For recompute, embeddings is the only thing requires gradient.
        if doing_recompute():
            return inputs_embeds

        if use_cache:
            warnings.warn("`use_cache` will set to False. Caching behavior is not supported in RoundPipe.")
        use_cache = False
        if past_key_values is not None:
            warnings.warn("`past_key_values` will be ignored. Caching behavior is not supported in RoundPipe.")
        past_key_values = None

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # It may already have been prepared by e.g. `generate`
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            # Prepare mask arguments
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            # Create the masks
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }
            # The sliding window alternating layers are not always activated depending on the config
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        return hidden_states, causal_mask_mapping, position_ids, position_embeddings, kwargs, labels, logits_to_keep

class Qwen3ForCausalLMWrappedLayer(nn.Module):
    def __init__(self, layer: nn.Module) -> None:
        super().__init__()
        self.layer = layer
    def forward(self, input):
        hidden_states, causal_mask_mapping, position_ids, position_embeddings, kwargs, labels, logits_to_keep = input
        hidden_states = self.layer(
            hidden_states,
            attention_mask=causal_mask_mapping[self.layer.attention_type],
            position_ids=position_ids,
            past_key_values=None,
            use_cache=False,
            cache_position=None,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        return hidden_states, causal_mask_mapping, position_ids, position_embeddings, kwargs, labels, logits_to_keep

class Qwen3ForCausalLMPostfix(nn.Module):
    def __init__(self, model: Qwen3ForCausalLM) -> None:
        super().__init__()
        self.norm = model.model.norm
        self.vocab_size = model.config.vocab_size
        self.lm_head = model.lm_head
        self.loss_function = model.loss_function

    def forward(self, input):
        hidden_states, causal_mask_mapping, position_ids, position_embeddings, kwargs, labels, logits_to_keep = input
        hidden_states = self.norm(hidden_states)
        
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
        )

EXPECTED_MODEL_CLASS = Qwen3ForCausalLM
def wrap_model(model: Qwen3ForCausalLM, **roundpipe_kwargs: Any) -> RoundPipe:
    model.loss_function = CompileForCausalLMLoss
    prefix = Qwen3ForCausalLMPrefix(model)
    layers = [Qwen3ForCausalLMWrappedLayer(layer) for layer in model.model.layers]
    postfix = Qwen3ForCausalLMPostfix(model)
    wrapped_model = RoundPipe(nn.Sequential(prefix, *layers, postfix), **roundpipe_kwargs)
    wrapped_model.set_original_model(model)
    return wrapped_model
