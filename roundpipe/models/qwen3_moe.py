from typing_extensions import *
import warnings
from importlib.metadata import version

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.masking_utils import (
    create_causal_mask,
    create_sliding_window_causal_mask,
)
from transformers.modeling_outputs import MoeCausalLMOutputWithPast
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeForCausalLM,
    Qwen3MoeDecoderLayer,
    Qwen3MoeSparseMoeBlock,
    load_balancing_loss_func,
)

transformers_version = tuple(map(int, version("transformers").split(".")[:2]))

from ..context import doing_recompute, save_for_recompute, get_recompute_data
from ..roundpipe import RoundPipe
from .function import CompileForCausalLMLoss


class Qwen3MoeForCausalLMPrefix(nn.Module):
    def __init__(self, model: Qwen3MoeForCausalLM) -> None:
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
            causal_mask, position_ids, position_embeddings = get_recompute_data()
            return (
                inputs_embeds,
                causal_mask,
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

        mask_function = (
            create_causal_mask
            if self.config.sliding_window is None
            else create_sliding_window_causal_mask
        )
        causal_mask = mask_function(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        hidden_states = inputs_embeds
        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        save_for_recompute(causal_mask, position_ids, position_embeddings)
        return (
            hidden_states,
            causal_mask,
            position_ids,
            position_embeddings,
            kwargs,
            labels,
            logits_to_keep,
            output_router_logits,
            attention_mask,
            [],  # router_logits
        )


class Qwen3MoeWrappedSparseMoeBlock(nn.Module):
    def __init__(self, mod: Qwen3MoeSparseMoeBlock):
        super().__init__()
        self.gate = mod.gate
        self.experts = mod.experts

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_reshaped = hidden_states.view(-1, hidden_dim)
        router_logits, routing_weights, selected_experts = self.gate(
            hidden_states_reshaped
        )
        final_hidden_states = self.experts(
            hidden_states_reshaped, selected_experts, routing_weights
        )
        return (
            final_hidden_states.reshape(batch_size, sequence_length, hidden_dim),
            router_logits,
        )


class Qwen3MoeOptSparseMoeBlock(nn.Module):
    def __init__(self, mod: Any) -> None:
        super().__init__()
        self.num_experts: int = mod.num_experts
        self.top_k: int = mod.top_k
        self.norm_topk_prob: bool = mod.norm_topk_prob

        # gating
        self.gate: nn.Module = mod.gate
        self.experts: nn.ModuleList = mod.experts

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(
            routing_weights, self.top_k, dim=-1
        )
        if self.norm_topk_prob:  # only diff with mixtral sparse moe block!
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        routing_weights = routing_weights.view(-1)
        selected_experts = selected_experts.view(-1)
        _, sort_idx = torch.sort(selected_experts)
        permute_weight = routing_weights[sort_idx]
        batch_idx = sort_idx.div(self.top_k, rounding_mode="floor")
        if doing_recompute():
            (token_per_expert_cpu,) = get_recompute_data()
        else:
            token_per_expert = torch.zeros(
                self.num_experts, dtype=torch.long, device=routing_weights.device
            )
            token_per_expert.index_add_(
                0, selected_experts, torch.ones_like(selected_experts, dtype=torch.long)
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
            expert_input = hidden_states[batch_idx[start_idx : start_idx + num_tokens]]
            expert_output = self.experts[expert_id](expert_input)
            expert_output *= permute_weight[
                start_idx : start_idx + num_tokens
            ].unsqueeze(-1)
            final_hidden_states.index_add_(
                0, batch_idx[start_idx : start_idx + num_tokens], expert_output
            )
            start_idx += num_tokens

        final_hidden_states = final_hidden_states.reshape(
            batch_size, sequence_length, hidden_dim
        )
        return final_hidden_states, router_logits


class Qwen3MoeForCausalLMWrappedLayer(nn.Module):
    def __init__(self, layer: Qwen3MoeDecoderLayer) -> None:
        super().__init__()
        self.hidden_size = layer.hidden_size
        self.self_attn = layer.self_attn
        if isinstance(layer.mlp, Qwen3MoeSparseMoeBlock):
            if transformers_version >= (5, 0):
                self.mlp = Qwen3MoeWrappedSparseMoeBlock(layer.mlp)
            else:
                self.mlp = Qwen3MoeOptSparseMoeBlock(layer.mlp)
        else:
            self.mlp = layer.mlp
        self.input_layernorm = layer.input_layernorm
        self.post_attention_layernorm = layer.post_attention_layernorm

    def forward(self, input):
        (
            hidden_states,
            causal_mask,
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
            position_embeddings=position_embeddings,
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_values=None,
            cache_position=None,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        # For the MoE layers, we need to unpack
        if isinstance(hidden_states, tuple):
            hidden_states, router_logit = hidden_states
            if output_router_logits:
                router_logits.append(router_logit)
        hidden_states = residual + hidden_states

        return (
            hidden_states,
            causal_mask,
            position_ids,
            position_embeddings,
            kwargs,
            labels,
            logits_to_keep,
            output_router_logits,
            attention_mask,
            router_logits,
        )


class Qwen3MoeForCausalLMPostfix(nn.Module):
    def __init__(self, model: Qwen3MoeForCausalLM) -> None:
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
            causal_mask,
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
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = cast(
                torch.Tensor,
                self.loss_function(logits, labels, self.vocab_size, **kwargs),
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


EXPECTED_MODEL_CLASS = Qwen3MoeForCausalLM


def wrap_model(model: Qwen3MoeForCausalLM, **roundpipe_kwargs: Any) -> RoundPipe:
    model.loss_function = CompileForCausalLMLoss
    prefix = Qwen3MoeForCausalLMPrefix(model)
    layers = [
        Qwen3MoeForCausalLMWrappedLayer(cast(Qwen3MoeDecoderLayer, layer))
        for layer in model.model.layers
    ]
    postfix = Qwen3MoeForCausalLMPostfix(model)
    wrapped_model = RoundPipe(
        nn.Sequential(prefix, *layers, postfix), **roundpipe_kwargs
    )
    wrapped_model.set_original_model(model)
    return wrapped_model
