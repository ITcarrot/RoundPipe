from beartype.typing import * # type: ignore[reportWildcardImportFromLibrary]
import threading
import heapq

import torch

from .threads import dump_all_active_threads, KeyboardInterruptRoundPipeThreads

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

        self.fwd_sem = [threading.Semaphore(0) for _ in self.fwd_plan]
        self.bwd_sem = [threading.Semaphore(0) for _ in self.bwd_plan]

    def backward_need_input(self, layer_id: int) -> bool:
        return any(r[0] == layer_id for r in self.bwd_plan)

    def forward_wait_for(self, layer_group_id: int) -> None:
        if layer_group_id < 0:
            return
        self.fwd_sem[layer_group_id].acquire()

    def forward_notify(self, layer_group_id: int) -> None:
        self.fwd_sem[layer_group_id].release()

    def forward_wait_complete(self, num_microbatch: int) -> None:
        if len(self.fwd_sem) == 0:
            return
        try:
            for _ in range(num_microbatch):
                self.fwd_sem[-1].acquire()
        except KeyboardInterrupt:
            dump_all_active_threads()
            raise KeyboardInterruptRoundPipeThreads from None

    def backward_wait_for(self, layer_group_id: int) -> None:
        if layer_group_id < 0:
            return
        self.bwd_sem[layer_group_id].acquire()

    def backward_notify(self, layer_group_id: int) -> None:
        self.bwd_sem[layer_group_id].release()

    def backward_wait_complete(self, num_microbatch: int) -> None:
        try:
            for _ in range(num_microbatch):
                self.bwd_sem[-1].acquire()
        except KeyboardInterrupt:
            dump_all_active_threads()
            raise KeyboardInterruptRoundPipeThreads from None

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

def chunk_layer_params(tensor_pair: List[Tuple[torch.Tensor, torch.Tensor]], n_chunks: int
                       ) -> List[List[Tuple[torch.Tensor, torch.Tensor]]]:
    def get_tensor_size(pair: Tuple[torch.Tensor, torch.Tensor]) -> int:
        src, dst = pair
        return src.numel() * src.element_size()
    tensor_pair.sort(key=get_tensor_size, reverse=True)
    chunk_scheme: List[List[Tuple[torch.Tensor, torch.Tensor]]] = [[] for _ in range(n_chunks)]
    chunk_heap = [(0, i) for i in range(n_chunks)]
    heapq.heapify(chunk_heap)
    for pair in tensor_pair:
        tensor_size = get_tensor_size(pair)
        cur_size, chunk_id = heapq.heappop(chunk_heap)
        chunk_scheme[chunk_id].append(pair)
        cur_size += tensor_size
        heapq.heappush(chunk_heap, (cur_size, chunk_id))
    sorted_scheme: List[List[Tuple[torch.Tensor, torch.Tensor]]] = []
    while chunk_heap:
        _, chunk_id = heapq.heappop(chunk_heap)
        sorted_scheme.append(chunk_scheme[chunk_id])
    return sorted_scheme
