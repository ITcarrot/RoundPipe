from typing import * # type: ignore[reportWildcardImportFromLibrary]
import traceback

import tqdm
import torch
import torch.nn as nn

from RoundPipe.batch import Batch
from RoundPipe.device import get_next_device
from RoundPipe.RunConfig import RoundPipeRunConfig, FullRoundPipeRunConfig
from RoundPipe.timer import ModelTimer
from RoundPipe.utils import get_model_size

class RoundPipe(nn.Module):
    def __init__(self,
                 model: nn.Module,
                 use_fp16: bool = False,
                 model_run_config: RoundPipeRunConfig = RoundPipeRunConfig()) -> None:
        super().__init__()
        filename, lineno, _, _ = traceback.extract_stack()[-2]
        self.name = f'{filename.split("/")[-1]}:{lineno}'
        self.model = model
        self.use_fp16 = use_fp16
        self.model_run_config = model_run_config

        if isinstance(model, nn.Sequential):
            self.layers = list(model)
        elif isinstance(model, (nn.ModuleList, nn.ModuleDict)):
            raise TypeError("RoundPipe does not support nn.ModuleList or nn.ModuleDict as the model directly. Please convert to nn.Sequential or wrap with nn.Module.")
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

    def forward(self, *args,
                roundpipe_run_config: RoundPipeRunConfig = RoundPipeRunConfig(), **kwargs) -> Any:
        full_run_config = FullRoundPipeRunConfig(roundpipe_run_config, self.model_run_config)
        hidden_states = Batch(args, kwargs, full_run_config)
        cur_layer = 0
        while cur_layer < self.num_layers:
            device = get_next_device()
            cur_layer = device.run(self, cur_layer, hidden_states, full_run_config)
        return hidden_states.dump(full_run_config)
