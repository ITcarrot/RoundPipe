from typing import * # type: ignore[reportWildcardImportFromLibrary]
import traceback

import torch
import torch.nn as nn

from RoundPipe.models import wrap_model
from RoundPipe.RoundPipe import RoundPipe
from RoundPipe.RunConfig import RoundPipeRunConfig
from RoundPipe.utils import get_model_size

def wrap_model_to_roundpipe(model: nn.Module,
                            use_sequential_preset: Optional[bool] = None,
                            lower_threshold: int = 16 * 1024,
                            upper_threshold: Optional[int] = None,
                            skip_modules: List[nn.Module] = [],
                            override_config: Dict[nn.Module, RoundPipeRunConfig] = {},
                            **roundpipe_kwargs: Any) -> nn.Module:
    if model in skip_modules:
        return model

    model_name = roundpipe_kwargs.get('name')
    if model_name is None:
        filename, lineno, _, _ = traceback.extract_stack()[-2]
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
        return RoundPipe(model, **roundpipe_kwargs)

    if upper_threshold is None:
        upper_threshold = get_model_size(model) // (torch.cuda.device_count() + 1)
    if (lower_threshold <= get_model_size(model, False) or get_model_size(model) <= upper_threshold) \
       and model.forward.__name__ != '_forward_unimplemented':
        if model in override_config:
            roundpipe_kwargs['model_run_config'] = override_config[model]
        elif get_model_size(model) < lower_threshold:
            modified_run_config = roundpipe_kwargs.get('model_run_config', RoundPipeRunConfig())
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
        modified_run_config: RoundPipeRunConfig = roundpipe_kwargs.get('model_run_config', RoundPipeRunConfig())
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
        return model
