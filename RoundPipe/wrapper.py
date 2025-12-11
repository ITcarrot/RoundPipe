"""Utilities that wrap user models with RoundPipe sequential presets."""

from beartype.typing import * # type: ignore[reportWildcardImportFromLibrary]
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
from .RoundPipe import RoundPipe
from .RunConfig import RoundPipeRunConfig
from .utils import get_model_size

@beartype
def wrap_model_to_roundpipe(model: nn.Module,
                            use_sequential_preset: Optional[bool] = None,
                            lower_threshold: int = 16 * 1024,
                            upper_threshold: Optional[int] = None,
                            skip_modules: List[nn.Module] = [],
                            override_config: Dict[nn.Module, RoundPipeRunConfig] = {},
                            **roundpipe_kwargs: Any) -> Union[nn.Module, RoundPipe]:
    """Recursively wrap modules in ``RoundPipe`` using heuristics/presets.

    Args:
        model: Root module to evaluate.
        use_sequential_preset: ``None``/``True`` attempts to replace with a
            packaged sequential preset, ``False`` skips the preset lookup.
        lower_threshold: Minimum size (bytes) to consider wrapping directly.
        upper_threshold: Maximum size before splitting submodules. Defaults to
            model size divided by ``num_devices + 1``.
        skip_modules: Modules that should remain untouched.
        override_config: Mapping from module objects to specific configs.
        **roundpipe_kwargs: Additional kwargs forwarded to ``RoundPipe``.

    Returns:
        Either the original module (possibly modified in-place) or a ``RoundPipe``
            instance wrapping the selected submodules.

    Raises:
        NotImplementedError: If ``use_sequential_preset`` is explicitly ``True``
            but no preset exists for the model type.
    """
    if model in skip_modules:
        return model

    model_name = roundpipe_kwargs.get('name')
    if model_name is None:
        filename, lineno, _, _ = traceback.extract_stack()[-3]
        model_name = f'{filename.split("/")[-1]}:{lineno}'

    try:
        if use_sequential_preset is None or use_sequential_preset:
            roundpipe_kwargs['name'] = model_name
            model = wrap_model(model, **roundpipe_kwargs)
            if use_sequential_preset is None:
                print(f'[INFO] Replace model "{model_name}" of type {model.__class__.__name__} with a sequential preset from RoundPipe.models')
                print(f'[INFO] If this is not expected, please call wrap_model_to_roundpipe with use_sequential_preset=False or rename the model class to avoid conflicts.')
            return model
    except NotImplementedError:
        if use_sequential_preset is True:
            raise
    if isinstance(model, nn.Sequential):
        roundpipe_kwargs['name'] = model_name
        if model in override_config:
            roundpipe_kwargs['model_run_config'] = override_config[model]
        return RoundPipe(model, **roundpipe_kwargs)

    if upper_threshold is None:
        upper_threshold = get_model_size(model) // (torch.cuda.device_count() + 1)
    if (lower_threshold <= get_model_size(model, False) or get_model_size(model) <= upper_threshold) \
       and model.forward.__name__ != '_forward_unimplemented':
        if model in override_config:
            roundpipe_kwargs['model_run_config'] = override_config[model]
        elif get_model_size(model) < lower_threshold:
            modified_run_config = copy.deepcopy(roundpipe_kwargs.get('model_run_config', RoundPipeRunConfig()))
            modified_run_config.num_microbatch = 1
            roundpipe_kwargs['model_run_config'] = modified_run_config
        roundpipe_kwargs['name'] = model_name
        return RoundPipe(model, **roundpipe_kwargs)
    elif isinstance(model, nn.ModuleList):
        if len(model) == 0:
            return model
        n_layers = len(model)
        roundpipe_kwargs['name'] = f'{model_name}.{n_layers - 1}'
        if model[n_layers - 1] in override_config:
            stash_config = roundpipe_kwargs.get('model_run_config', RoundPipeRunConfig())
            roundpipe_kwargs['model_run_config'] = override_config[model[n_layers - 1]]
            model[n_layers - 1] = RoundPipe(model[n_layers - 1], **roundpipe_kwargs)
            roundpipe_kwargs['model_run_config'] = stash_config
        else:
            model[n_layers - 1] = RoundPipe(model[n_layers - 1], **roundpipe_kwargs)
        # do not merge output at non-final layers
        modified_run_config: RoundPipeRunConfig = copy.deepcopy(roundpipe_kwargs.get('model_run_config', RoundPipeRunConfig()))
        modified_run_config.merge_output = False
        roundpipe_kwargs['model_run_config'] = modified_run_config
        for layer_idx in range(n_layers - 1):
            roundpipe_kwargs['name'] = f'{model_name}.{layer_idx}'
            if model[layer_idx] in override_config:
                stash_config = roundpipe_kwargs.get('model_run_config', RoundPipeRunConfig())
                roundpipe_kwargs['model_run_config'] = override_config[model[layer_idx]]
                model[layer_idx] = RoundPipe(model[layer_idx], **roundpipe_kwargs)
                roundpipe_kwargs['model_run_config'] = stash_config
            else:
                model[layer_idx] = RoundPipe(model[layer_idx], **roundpipe_kwargs)
        return model
    else:
        for submodel_name, submodel in model.named_children():
            roundpipe_kwargs['name'] = f'{model_name}.{submodel_name}'
            wrapped_submodel = wrap_model_to_roundpipe(submodel, 
                                                       use_sequential_preset,
                                                       lower_threshold,
                                                       upper_threshold,
                                                       skip_modules,
                                                       override_config,
                                                       **roundpipe_kwargs)
            if wrapped_submodel is not submodel:
                setattr(model, submodel_name, wrapped_submodel)

        # Override loss function for HuggingFace models
        # Check only when this layer is not wrapped as a hole
        if HFPreTrainedModel is not None and isinstance(model, HFPreTrainedModel):
            roundpipe_kwargs['name'] = model_name + '.loss_function'
            model._loss_function = wrap_model(model.loss_function, **roundpipe_kwargs)

        return model
