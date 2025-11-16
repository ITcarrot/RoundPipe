from beartype.typing import * # type: ignore[reportWildcardImportFromLibrary]

import torch

if TYPE_CHECKING:
    from RoundPipe import RoundPipe
else:
    from typing_extensions import TypeAliasType
    RoundPipe = TypeAliasType('RoundPipe', 'RoundPipe.RoundPipe')

class ModelExecutePlan:
    def __init__(self, model: 'RoundPipe', fuse_fwd_bwd: bool) -> None:
        self.fwd_plan: List[range] = []
        self.bwd_plan: List[range] = []

        if fuse_fwd_bwd:
            self.fwd_plan = [range(i, i + 1) for i in range(model.num_layers - 1)]
            self.bwd_plan = [range(i, i + 1) for i in reversed(range(model.num_layers))]
        else:
            self.fwd_plan = [range(i, i + 1) for i in range(model.num_layers)]
            self.bwd_plan = list(reversed(self.fwd_plan))

    def backward_need_input(self, layer_id: int) -> bool:
        return any(r[0] == layer_id for r in self.bwd_plan)

class BackwardScheduleSimulator:
    def __init__(self):
        self.tags = [torch.tensor(0., dtype=torch.float32, requires_grad=True) for _ in range(torch.cuda.device_count())]
        self.cur_device = 0
        self.n_devices = torch.cuda.device_count()
    
    def get_next_tag(self) -> torch.Tensor:
        self.cur_device = (self.cur_device + 1) % self.n_devices
        tag = self.tags[self.cur_device]
        return tag
    
    def update_current_tag(self, new_tag: torch.Tensor) -> None:
        self.tags[self.cur_device] = new_tag
    
    def reset(self) -> None:
        self.cur_device = 0
        self.tags = [torch.tensor(0., dtype=torch.float32, requires_grad=True) for _ in range(torch.cuda.device_count())]

backward_schedule_simulator = BackwardScheduleSimulator()
