"""Model wrappers for RoundPipe-supported models.

This module provides utilities to wrap supported model architectures
with RoundPipe's sequential execution presets. Each supported model type
has a corresponding wrapper module that implements the necessary integration.

Attributes:
    DISABLE_TORCH_COMPILE (bool): flag indicating whether to disable torch.compile in models.
    SUPPORTED_MODELS (Dict[str, str]): dictionary mapping model type names
        to their corresponding wrapper module paths.
"""

from typing_extensions import *
import importlib

import torch

from ..device import get_num_devices
from ..roundpipe import RoundPipe

DISABLE_TORCH_COMPILE = False
if not DISABLE_TORCH_COMPILE:
    torch._dynamo.config.cache_size_limit *= get_num_devices()
    torch._dynamo.config.accumulated_cache_size_limit *= get_num_devices()

SUPPORTED_MODELS = {
    "function": ".function",
    "LlamaForCausalLM": ".llama",
    "Qwen3MoeForCausalLM": ".qwen3_moe",
    "Qwen3ForCausalLM": ".qwen3",
}


def wrap_model(model: Any, **roundpipe_kwargs: Any) -> RoundPipe:
    """Wrap a supported model with RoundPipe's sequential preset.

    Args:
        model: The model instance to be wrapped.
        **roundpipe_kwargs: Additional keyword arguments specific to the
            model wrapper.

    Returns:
        A RoundPipe model instance.

    Raises:
        NotImplementedError: If the model type is not supported or if the
            model class does not match the expected class for its type.
    """
    model_type = type(model).__name__
    if model_type in SUPPORTED_MODELS:
        module = importlib.import_module(
            SUPPORTED_MODELS[model_type], package=__package__
        )
        expected_class = getattr(module, "EXPECTED_MODEL_CLASS")
        if not isinstance(model, expected_class):
            raise NotImplementedError(
                f"Model class {type(model)} is not an instance of expected class {expected_class}."
            )
        wrap_func = getattr(module, "wrap_model")
        return wrap_func(model, **roundpipe_kwargs)
    else:
        raise NotImplementedError(
            f"Model type {model_type} does not have a sequential preset."
        )


__all__ = ["wrap_model"]
