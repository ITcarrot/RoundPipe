from typing import * # type: ignore[reportWildcardImportFromLibrary]
import traceback
import inspect

import tqdm
import torch
import torch.nn as nn

from RoundPipe.batch import Batch
from RoundPipe.device import get_next_device
from RoundPipe.models import wrap_model
from RoundPipe.RunConfig import RoundPipeRunConfig, FullRoundPipeRunConfig
from RoundPipe.timer import ModelTimer
from RoundPipe.utils import get_model_size

class RoundPipe(nn.Module):
    def __init__(self,
                 model: nn.Module,
                 use_fp16: bool = False,
                 name: Optional[str] = None,
                 model_run_config: RoundPipeRunConfig = RoundPipeRunConfig()) -> None:
        super().__init__()
        filename, lineno, _, _ = traceback.extract_stack()[-2]
        self.name = name if name else f'{filename.split("/")[-1]}:{lineno}'
        self.model = model
        self.original_model: Optional[nn.Module] = None # placeholder for original model if needing its functions
        self.use_fp16 = use_fp16
        self.model_run_config = model_run_config

        if isinstance(model, nn.Sequential):
            self.layers = list(model)
        else:
            self.layers = [model]

        self.num_layers = len(self.layers)
        self.layer_workload: List[float] = []
        self.layer_has_param: List[bool] = []
        for layer in self.layers:
            self.layer_has_param.append(any(True for _ in layer.parameters()))
            self.layer_workload.append(get_model_size(layer))
        self.model_timer = ModelTimer(self.num_layers)

        print(f'Processing parameters in RoundPipe model "{self.name}" ...')
        for parm in tqdm.tqdm(self.model.parameters(), total=sum(1 for _ in self.model.parameters())):
            pinned_tensor = torch.empty_like(parm.data, dtype=torch.float16 if use_fp16 and parm.is_floating_point() else None, pin_memory=True)
            pinned_tensor.copy_(parm.data)
            parm.data = pinned_tensor
            parm.data_cpu = pinned_tensor # type: ignore[attr-defined]
        print(f'Processing buffers in RoundPipe model "{self.name}" ...')
        for buffer in tqdm.tqdm(self.model.buffers(), total=sum(1 for _ in self.model.buffers())):
            pinned_tensor = torch.empty_like(buffer.data, dtype=torch.float16 if use_fp16 and buffer.is_floating_point() else None, pin_memory=True)
            pinned_tensor.copy_(buffer.data)
            buffer.data = pinned_tensor
            buffer.data_cpu = pinned_tensor # type: ignore[attr-defined]

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            if self.original_model is not None:
                return getattr(self.original_model, name)
            return getattr(self.model, name)

    def forward(self, *args,
                roundpipe_run_config: RoundPipeRunConfig = RoundPipeRunConfig(), **kwargs) -> Any:
        full_run_config = FullRoundPipeRunConfig(roundpipe_run_config, self.model_run_config)
        hidden_states = Batch(args, kwargs, full_run_config)
        cur_layer = 0
        while cur_layer < self.num_layers:
            device = get_next_device()
            cur_layer = device.run(self, cur_layer, hidden_states, full_run_config)
        return hidden_states.dump(full_run_config)

def wrap_model_to_roundpipe(model: nn.Module,
                            wrap_threshold: Optional[int] = None,
                            **roundpipe_kwargs: Any) -> nn.Module:
    model_name = roundpipe_kwargs.get('name')
    if model_name is None:
        filename, lineno, _, _ = traceback.extract_stack()[-2]
        model_name = f'{filename.split("/")[-1]}:{lineno}'

    try:
        roundpipe_kwargs['name'] = model_name
        return wrap_model(model, **roundpipe_kwargs)
    except NotImplementedError:
        pass
    if isinstance(model, nn.Sequential):
        roundpipe_kwargs['name'] = model_name
        return RoundPipe(model, **roundpipe_kwargs)

    if hasattr(model, 'loss_function') and inspect.isfunction(model.loss_function):
        roundpipe_kwargs['name'] = f'{model_name}.loss_function'
        model.loss_function = wrap_model(model.loss_function, **roundpipe_kwargs)

    if wrap_threshold is None:
        wrap_threshold = get_model_size(model) // (torch.cuda.device_count() + 1)
    if get_model_size(model) <= wrap_threshold and model.forward.__name__ != '_forward_unimplemented':
        modified_run_config: RoundPipeRunConfig = roundpipe_kwargs.get('model_run_config', RoundPipeRunConfig())
        modified_run_config.num_microbatch = 1
        roundpipe_kwargs['model_run_config'] = modified_run_config
        roundpipe_kwargs['name'] = model_name
        return RoundPipe(model, **roundpipe_kwargs)
    elif isinstance(model, nn.ModuleList):
        if len(model) == 0:
            return model
        n_layers = len(model)
        roundpipe_kwargs['name'] = f'{model_name}.{n_layers - 1}'
        model[n_layers - 1] = RoundPipe(model[n_layers - 1], **roundpipe_kwargs)
        # do not merge output at non-final layers
        modified_run_config: RoundPipeRunConfig = roundpipe_kwargs.get('model_run_config', RoundPipeRunConfig())
        modified_run_config.merge_output = False
        roundpipe_kwargs['model_run_config'] = modified_run_config
        for layer_idx in range(n_layers - 1):
            roundpipe_kwargs['name'] = f'{model_name}.{layer_idx}'
            model[layer_idx] = RoundPipe(model[layer_idx], **roundpipe_kwargs)
        return model
    else:
        for submodel_name, submodel in model.named_children():
            roundpipe_kwargs['name'] = f'{model_name}.{submodel_name}'
            wrapped_submodel = wrap_model_to_roundpipe(submodel, wrap_threshold, **roundpipe_kwargs)
            if wrapped_submodel is not submodel:
                setattr(model, submodel_name, wrapped_submodel)
        return model
