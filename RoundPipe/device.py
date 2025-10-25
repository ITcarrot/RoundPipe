from typing import *

import torch

from RoundPipe.profile import annotate
from RoundPipe.run import CheckpointRun

MERGE_LAYER_FACTOR = 1.1

if TYPE_CHECKING:
    from RoundPipe.batch import Batch
    from RoundPipe.RoundPipe import RoundPipe
    from RoundPipe.RunConfig import FullRoundPipeRunConfig

class DeviceManager:
    def __init__(self, device: torch.device):
        self.device = device
        
        self.upstream: torch.cuda.Stream = torch.cuda.Stream(device) # type: ignore[reportAttributeAccessIssue]
        self.compute_stream = torch.cuda.default_stream(self.device)
        self.downstream: torch.cuda.Stream = torch.cuda.Stream(device) # type: ignore[reportAttributeAccessIssue]
        
        self.order_tag = torch.empty(0, requires_grad = True)
    
    def run(self, model: 'RoundPipe', cur_layer: int, batch: 'Batch', run_config: 'FullRoundPipeRunConfig') -> int:
        layer_start = cur_layer
        workload_to_process = model.layer_workload[layer_start]
        layer_to_process = 1
        layer_ids = range(layer_start, layer_start + layer_to_process)
        layer_require_grad = any(model.layer_has_param[layer_start + i] for i in range(layer_to_process))
        for i in range(run_config.num_microbatch):
            input_requrie_grad = any(isinstance(t, torch.Tensor) and t.requires_grad for t in batch.flatten_states[i])
            with annotate(f'{model.name}L[{layer_start}, {layer_start + layer_to_process})B{i}'):
                with torch.enable_grad() if run_config.requires_grad else torch.no_grad():
                    order_tag = self.order_tag if layer_require_grad or input_requrie_grad else torch.empty(0)
                    order_tag, *batch.flatten_states[i] \
                        = CheckpointRun.apply(self, model, run_config, batch, layer_ids, i, order_tag, *batch.flatten_states[i]) # type: ignore[reportGeneralTypeIssues]
                    if order_tag.requires_grad:
                        self.order_tag = order_tag
        return layer_start + layer_to_process

device_list: List[DeviceManager] = []
cur_device = 0

for i in range(torch.cuda.device_count()):
    device = DeviceManager(torch.device(f"cuda:{i}"))
    device_list.append(device)

def get_next_device() -> DeviceManager:
    global cur_device
    device = device_list[cur_device]
    cur_device = (cur_device + 1) % len(device_list)
    return device

def device_tag_detach():
    for device in device_list:
        device.order_tag = torch.empty(0, requires_grad = True)
