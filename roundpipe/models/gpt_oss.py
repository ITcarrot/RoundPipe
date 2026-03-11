from typing_extensions import *
import warnings

import torch
import torch.nn as nn
from transformers.masking_utils import (
    create_causal_mask,
    create_sliding_window_causal_mask,
)
from transformers.modeling_outputs import MoeCausalLMOutputWithPast
from transformers.models.gpt_oss.modeling_gpt_oss import (
    GptOssForCausalLM,
    GptOssDecoderLayer,
    GptOssExperts,
    load_balancing_loss_func,
)

from ..context import doing_recompute, save_for_recompute, get_recompute_data
from ..roundpipe import RoundPipe
from .function import CompileForCausalLMLoss, ChunkedCompileLinearForCausalLMLoss


class GptOssOptExperts(nn.Module):
    def __init__(self, mod: GptOssExperts) -> None:
        super().__init__()
        self.num_experts = mod.num_experts
        self.hidden_size = mod.hidden_size
        self.alpha = mod.alpha
        self.limit = mod.limit

        self.gate_up_proj = mod.gate_up_proj
        self.gate_up_proj_bias = mod.gate_up_proj_bias
        self.down_proj = mod.down_proj
        self.down_proj_bias = mod.down_proj_bias

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_indices: torch.Tensor,
        routing_weights: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        top_k = router_indices.shape[-1]
        selected_experts = router_indices.view(-1)
        routing_weights = torch.gather(routing_weights, 1, router_indices)
        routing_weights = routing_weights.view(-1)

        _, sort_idx = torch.sort(selected_experts)
        permute_weight = routing_weights[sort_idx]
        batch_idx = sort_idx.div(top_k, rounding_mode="floor")

        if doing_recompute():
            (token_per_expert_cpu,) = get_recompute_data()
        else:
            token_per_expert = torch.zeros(
                self.num_experts, dtype=torch.long, device=hidden_states.device
            )
            token_per_expert.index_add_(
                0,
                selected_experts,
                torch.ones_like(selected_experts, dtype=torch.long),
            )
            token_per_expert_cpu = token_per_expert.cpu().numpy()
            save_for_recompute(token_per_expert_cpu)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        start_idx = 0
        for expert_id in range(self.num_experts):
            num_tokens = token_per_expert_cpu[expert_id]
            if num_tokens == 0:
                continue
            expert_tokens = batch_idx[start_idx : start_idx + num_tokens]
            expert_input = hidden_states[expert_tokens]

            gate_up = (
                expert_input @ self.gate_up_proj[expert_id]
                + self.gate_up_proj_bias[expert_id]
            )
            gate, up = gate_up[..., ::2], gate_up[..., 1::2]
            gate = gate.clamp(min=None, max=self.limit)
            up = up.clamp(min=-self.limit, max=self.limit)
            glu = gate * torch.sigmoid(gate * self.alpha)
            gated_output = (up + 1) * glu
            expert_output = (
                gated_output @ self.down_proj[expert_id]
                + self.down_proj_bias[expert_id]
            )
            expert_output *= permute_weight[
                start_idx : start_idx + num_tokens
            ].unsqueeze(-1)
            final_hidden_states.index_add_(0, expert_tokens, expert_output)
            start_idx += num_tokens

        return final_hidden_states.view(batch_size, sequence_length, hidden_dim)


class GptOssForCausalLMPrefix(nn.Module):
    def __init__(self, model: GptOssForCausalLM) -> None:
        super().__init__()
        self.embed_tokens = model.model.embed_tokens
        self.rotary_emb = model.model.rotary_emb
        self.config = model.model.config

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Any] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        cache_position: Optional[torch.Tensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Any,
    ):
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        if inputs_embeds is None:
            inputs_embeds = cast(torch.Tensor, self.embed_tokens(input_ids))

        if output_router_logits is None:
            output_router_logits = self.config.output_router_logits

        if doing_recompute():
            causal_mask_mapping, position_ids, position_embeddings = (
                get_recompute_data()
            )
            return (
                inputs_embeds,
                causal_mask_mapping,
                position_ids,
                position_embeddings,
                kwargs,
                labels,
                logits_to_keep,
                output_router_logits,
                attention_mask,
                [],  # router_logits
            )

        if use_cache:
            warnings.warn(
                "`use_cache` will set to False. Caching behavior is not supported in RoundPipe."
            )
        use_cache = False
        if past_key_values is not None:
            warnings.warn(
                "`past_key_values` will be ignored. Caching behavior is not supported in RoundPipe."
            )
        past_key_values = None

        if cache_position is None:
            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        if not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
            }
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
                "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
            }

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        save_for_recompute(causal_mask_mapping, position_ids, position_embeddings)
        return (
            hidden_states,
            causal_mask_mapping,
            position_ids,
            position_embeddings,
            kwargs,
            labels,
            logits_to_keep,
            output_router_logits,
            attention_mask,
            [],  # router_logits
        )


class GptOssForCausalLMWrappedLayer(nn.Module):
    def __init__(self, layer: GptOssDecoderLayer) -> None:
        super().__init__()
        self.hidden_size = layer.hidden_size
        self.self_attn = layer.self_attn
        self.mlp = layer.mlp
        self.input_layernorm = layer.input_layernorm
        self.post_attention_layernorm = layer.post_attention_layernorm
        self.attention_type = layer.attention_type

    def forward(self, input):
        (
            hidden_states,
            causal_mask_mapping,
            position_ids,
            position_embeddings,
            kwargs,
            labels,
            logits_to_keep,
            output_router_logits,
            attention_mask,
            router_logits,
        ) = input

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=causal_mask_mapping[self.attention_type],
            position_ids=position_ids,
            past_key_values=None,
            use_cache=False,
            cache_position=None,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states, router_logit = self.mlp(hidden_states)
        if output_router_logits:
            router_logits.append(router_logit)
        hidden_states = residual + hidden_states

        return (
            hidden_states,
            causal_mask_mapping,
            position_ids,
            position_embeddings,
            kwargs,
            labels,
            logits_to_keep,
            output_router_logits,
            attention_mask,
            router_logits,
        )


class GptOssForCausalLMPostfix(nn.Module):
    def __init__(self, model: GptOssForCausalLM) -> None:
        super().__init__()
        self.norm = model.model.norm
        self.vocab_size = model.config.vocab_size
        self.lm_head = model.lm_head
        self.loss_function = model.loss_function

        self.num_experts: int = model.num_experts
        self.num_experts_per_tok: int = model.num_experts_per_tok
        self.router_aux_loss_coef: float = model.router_aux_loss_coef

    def forward(self, input) -> MoeCausalLMOutputWithPast:
        (
            hidden_states,
            causal_mask_mapping,
            position_ids,
            position_embeddings,
            kwargs,
            labels,
            logits_to_keep,
            output_router_logits,
            attention_mask,
            router_logits,
        ) = input
        hidden_states = self.norm(hidden_states)

        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )
        logits = None
        if kwargs.get("return_logits", True):
            logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            if logits is None:
                loss = ChunkedCompileLinearForCausalLMLoss(
                    hidden_states[:, slice_indices, :],
                    self.lm_head,
                    labels,
                    **kwargs,
                )
            else:
                loss = self.loss_function(
                    logits=logits, labels=labels, vocab_size=self.vocab_size, **kwargs
                )

        aux_loss = None
        if output_router_logits:
            aux_loss = cast(
                torch.FloatTensor,
                load_balancing_loss_func(
                    tuple(t.float() for t in router_logits),
                    self.num_experts,
                    self.num_experts_per_tok,
                    attention_mask,
                ),
            )
            if loss is not None:
                loss += self.router_aux_loss_coef * aux_loss.to(
                    loss.device
                )  # make sure to reside in the same device

        return MoeCausalLMOutputWithPast(
            loss=cast(Optional[torch.FloatTensor], loss),
            aux_loss=aux_loss,
            logits=logits,
            router_logits=router_logits,
        )


EXPECTED_MODEL_CLASS = GptOssForCausalLM


def wrap_model(model: GptOssForCausalLM, **roundpipe_kwargs: Any) -> RoundPipe:
    model.loss_function = CompileForCausalLMLoss

    for layer in model.model.layers:
        layer = cast(GptOssDecoderLayer, layer)
        layer.mlp.experts = cast(GptOssExperts, GptOssOptExperts(layer.mlp.experts))

    prefix = GptOssForCausalLMPrefix(model)
    layers = [
        GptOssForCausalLMWrappedLayer(cast(GptOssDecoderLayer, layer))
        for layer in model.model.layers
    ]
    postfix = GptOssForCausalLMPostfix(model)
    wrapped_model = RoundPipe(
        nn.Sequential(prefix, *layers, postfix), **roundpipe_kwargs
    )
    wrapped_model.set_original_model(model)
    return wrapped_model
