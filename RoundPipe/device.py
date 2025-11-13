from typing import * # type: ignore[reportWildcardImport]

import torch

from RoundPipe.run import run_forward, run_backward

MERGE_LAYER_FACTOR = 1.1

if TYPE_CHECKING:
    from RoundPipe.batch import Batch
    from RoundPipe.run import RoundPipeRunContext
    from RoundPipe.RoundPipe import RoundPipe
    from RoundPipe.RunConfig import FullRoundPipeRunConfig

class DeviceManager:
    def __init__(self, id: int, device: torch.device):
        self.id = id
        self.device = device
        
        self.upstream: torch.cuda.Stream = torch.cuda.Stream(device) # type: ignore[reportAttributeAccessIssue]
        self.compute_stream = torch.cuda.default_stream(self.device)
        self.downstream: torch.cuda.Stream = torch.cuda.Stream(device) # type: ignore[reportAttributeAccessIssue]

    def launch_forward(self, layer_ids: range, batch: 'Batch',
                       run_context: 'List[RoundPipeRunContext]') -> None:
        for batch_context in run_context:
            run_forward(layer_ids, batch, batch_context, self)
    
    def launch_backward(self, layer_ids: range,
                        run_context: 'List[RoundPipeRunContext]') -> None:
        for batch_context in run_context:
            run_backward(layer_ids, batch_context, self)

device_list: List[DeviceManager] = []
cur_device = 0

for i in range(torch.cuda.device_count()):
    device = DeviceManager(i, torch.device(f"cuda:{i}"))
    device_list.append(device)

def get_next_device() -> DeviceManager:
    global cur_device
    device = device_list[cur_device]
    cur_device = (cur_device + 1) % len(device_list)
    return device
