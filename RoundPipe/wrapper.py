"""Utilities that wrap user models with RoundPipe sequential presets."""

from beartype.typing import * # pyright: ignore[reportWildcardImportFromLibrary]
from beartype import beartype
import traceback
import copy

import torch
import torch.nn as nn
try:
    from transformers.modeling_utils import PreTrainedModel as HFPreTrainedModel
except ImportError:
    HFPreTrainedModel = None

from .models import wrap_model
from .RoundPipe import RoundPipe, AutoRoundPipe
from .RunConfig import RoundPipeRunConfig
from .utils import get_model_size

def wrap_model_recursive(model: nn.Module,
                         lower_threshold: int,
                         upper_threshold: int,
                         skip_modules: Container[nn.Module],
                         override_config: Dict[nn.Module, RoundPipeRunConfig],
                         model_run_config: RoundPipeRunConfig,
                         name: str,
                         **roundpipe_kwargs: Any) -> nn.Module:
    """Recursively wrap modules in ``RoundPipe`` using heuristics/presets.

    Args:
        model: Root module to evaluate.
        lower_threshold: Minimum size (bytes) to consider wrapping directly.
        upper_threshold: Maximum size before splitting submodules. Defaults to
            model size divided by ``num_devices + 1``.
        skip_modules: Modules that should remain untouched.
        override_config: Mapping from module objects to specific configs.
        model_run_config: Default run config for ``RoundPipe`` instances.
        name: Name of the current module.
        **roundpipe_kwargs: Additional kwargs forwarded to ``RoundPipe``.

    Returns:
        Either the original module (possibly modified in-place) or a ``RoundPipe``
            instance wrapping the selected submodules.
    """
    if model in skip_modules:
        return model

    if isinstance(model, nn.Sequential):
        return RoundPipe(model, name=name,
                         model_run_config=override_config.get(model, model_run_config),
                         **roundpipe_kwargs)

    if (lower_threshold <= get_model_size(model, False) or get_model_size(model) <= upper_threshold) \
       and model.forward.__name__ != '_forward_unimplemented':
        if model in override_config:
            model_run_config = override_config[model]
        elif get_model_size(model) < lower_threshold:
            model_run_config = copy.deepcopy(model_run_config)
            model_run_config.num_microbatch = 1
        return RoundPipe(model, name=name, model_run_config=model_run_config,
                         **roundpipe_kwargs)
    elif isinstance(model, nn.ModuleList):
        if len(model) == 0:
            return model
        n_layers = len(model)
        model[n_layers - 1] = RoundPipe(
            model[n_layers - 1], name=f'{name}.{n_layers - 1}',
            model_run_config=override_config.get(model[n_layers - 1], model_run_config),
            **roundpipe_kwargs
        )
        # do not merge output at non-final layers
        model_run_config = copy.deepcopy(model_run_config)
        model_run_config.merge_output = False
        for layer_idx in range(n_layers - 1):
            model[layer_idx] = RoundPipe(
                model[layer_idx], name=f'{name}.{layer_idx}',
                model_run_config=override_config.get(model[layer_idx], model_run_config),
                **roundpipe_kwargs
            )
        return model
    else:
        for submodel_name, submodel in model.named_children():
            wrapped_submodel = wrap_model_recursive(
                submodel, lower_threshold, upper_threshold, skip_modules,
                override_config, model_run_config,
                f'{name}.{submodel_name}', **roundpipe_kwargs
            )
            if wrapped_submodel is not submodel:
                setattr(model, submodel_name, wrapped_submodel)

        # Override loss function for HuggingFace models
        # Check only when this layer is not wrapped as a hole
        if HFPreTrainedModel is not None and isinstance(model, HFPreTrainedModel):
            model._loss_function = wrap_model(
                model.loss_function, name=f'{name}.loss_function',
                model_run_config=model_run_config, **roundpipe_kwargs
            )

        return model

def wrap_model_to_roundpipe(model: nn.Module,
                            use_sequential_preset: Optional[bool] = None,
                            lower_threshold: int = 16 * 1024,
                            upper_threshold: Optional[int] = None,
                            skip_modules: Container[nn.Module] = [],
                            override_config: Dict[nn.Module, RoundPipeRunConfig] = {},
                            model_run_config: RoundPipeRunConfig = RoundPipeRunConfig(),
                            name: Optional[str] = None,
                            **roundpipe_kwargs: Any) -> Union[RoundPipe, AutoRoundPipe]:
    """Wrap a model into RoundPipe instance using recursive heuristics/presets.

    Args:
        model: Root module to evaluate.
        use_sequential_preset: ``None``/``True`` attempts to replace with a
            packaged sequential preset, ``False`` skips the preset lookup.
        lower_threshold: Minimum size (bytes) to consider wrapping directly.
        upper_threshold: Maximum size before splitting submodules. Defaults to
            model size divided by ``num_devices + 1``.
        skip_modules: Modules that should remain untouched.
        override_config: Mapping from module objects to specific configs.
        model_run_config: Default run config for ``RoundPipe`` instances.
        name: Name of the current module. If ``None``, a name is generated
            based on the call site.
        **roundpipe_kwargs: Additional kwargs forwarded to ``RoundPipe``.

    Returns:
        A RoundPipe managed model instance wrapping the selected submodules. 

    Raises:
        NotImplementedError: If ``use_sequential_preset`` is explicitly ``True``
            but no preset exists for the model type.
    """
    if name is None:
        filename, lineno, _, _ = traceback.extract_stack()[-3]
        name = f'{filename.split("/")[-1]}:{lineno}'

    try:
        if use_sequential_preset is None or use_sequential_preset:
            model = wrap_model(model, name=name, model_run_config=model_run_config,
                               **roundpipe_kwargs)
            if use_sequential_preset is None:
                print(f'[INFO] Replace model "{name}" of type {model.__class__.__name__} with a sequential preset from RoundPipe.models')
                print(f'[INFO] If this is not expected, please call wrap_model_to_roundpipe with use_sequential_preset=False or rename the model class to avoid conflicts.')
            return model
    except NotImplementedError:
        if use_sequential_preset is True:
            raise

    if upper_threshold is None:
        upper_threshold = get_model_size(model) // (torch.cuda.device_count() + 1)
    wrapped_model = wrap_model_recursive(
        model, lower_threshold, upper_threshold, skip_modules,
        override_config, model_run_config, name, **roundpipe_kwargs
    )
    if isinstance(wrapped_model, RoundPipe):
        return wrapped_model
    else:
        return AutoRoundPipe(wrapped_model, name=name, **roundpipe_kwargs)
