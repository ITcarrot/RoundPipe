"""Scheduling utilities that orchestrate forward/backward ordering.

Attributes:
    backward_schedule_simulator: Global simulator for pipelined backward scheduling.
"""

from typing_extensions import *
import threading
import heapq

import torch

from .threads import dump_all_active_threads, KeyboardInterruptRoundPipeThreads

if TYPE_CHECKING:
    from .roundpipe import RoundPipe
else:
    RoundPipe = TypeAliasType("RoundPipe", "roundpipe.roundpipe.RoundPipe")


class ModelExecutePlan:
    """Encodes layer grouping and synchronization for forward/backward.

    Attributes:
        fwd_plan: List of layer ranges executed during forward.
        bwd_plan: List of layer ranges executed during backward.
        fwd_sem: Per-layer semaphores used to gate forward progress.
        bwd_sem: Per-layer semaphores used to gate backward progress.
    """

    def __init__(self, model: "RoundPipe", fuse_fwd_bwd: bool) -> None:
        """Initialize execution plans based on the model configuration.

        Args:
            model: ``RoundPipe`` wrapper whose layers are being scheduled.
            fuse_fwd_bwd: Whether forward/backward stages are fused.
        """
        self.fwd_plan: List[range] = []
        self.bwd_plan: List[range] = []

        if fuse_fwd_bwd:
            self.fwd_plan = [range(i, i + 1) for i in range(model.num_layers - 1)]
            self.bwd_plan = [range(i, i + 1) for i in reversed(range(model.num_layers))]
        else:
            self.fwd_plan = [range(i, i + 1) for i in range(model.num_layers)]
            self.bwd_plan = list(reversed(self.fwd_plan))

        self.fwd_sem: List[threading.Semaphore] = [
            threading.Semaphore(0) for _ in self.fwd_plan
        ]
        self.bwd_sem: List[threading.Semaphore] = [
            threading.Semaphore(0) for _ in self.bwd_plan
        ]

    def backward_need_input(self, layer_id: int) -> bool:
        """Return whether backward execution requires inputs for ``layer_id``."""
        return any(r[0] == layer_id for r in self.bwd_plan)

    def forward_wait_for(self, layer_group_id: int) -> None:
        """Block until the previous forward group finishes.

        Args:
            layer_group_id: Index of the group to wait on minus one.
        """
        if layer_group_id < 0:
            return
        self.fwd_sem[layer_group_id].acquire()

    def forward_notify(self, layer_group_id: int) -> None:
        """Signal that the given forward group completed."""
        self.fwd_sem[layer_group_id].release()

    def forward_wait_complete(self, num_microbatch: int) -> None:
        """Wait for the last forward group to finish ``num_microbatch`` times.

        Args:
            num_microbatch: Number of microbatches that must finish.
        """
        if len(self.fwd_sem) == 0:
            return
        try:
            for _ in range(num_microbatch):
                self.fwd_sem[-1].acquire()
        except KeyboardInterrupt:
            dump_all_active_threads()
            raise KeyboardInterruptRoundPipeThreads from None

    def backward_wait_for(self, layer_group_id: int) -> None:
        """Block until the given backward group completes."""
        if layer_group_id < 0:
            return
        self.bwd_sem[layer_group_id].acquire()

    def backward_notify(self, layer_group_id: int) -> None:
        """Signal that the given backward group completed."""
        self.bwd_sem[layer_group_id].release()

    def backward_wait_complete(self, num_microbatch: int) -> None:
        """Wait for backward completion across all microbatches.

        Args:
            num_microbatch: Number of microbatches that must finish.
        """
        try:
            for _ in range(num_microbatch):
                self.bwd_sem[-1].acquire()
        except KeyboardInterrupt:
            dump_all_active_threads()
            raise KeyboardInterruptRoundPipeThreads from None


class BackwardScheduleSimulator:
    """Mimics async backward tagging to coordinate microbatch ordering.

    This class simulates the behavior of pipelined backward scheduling by
    maintaining a set of gradient anchor tensors (tags) that are rotated among
    devices. Each device uses its assigned tag to build the autograd graph for
    its microbatch during backward passes. By rotating the tags, we ensure that
    the backward passes across microbatches and layers are scheduled as desired.

    Attributes:
        tags: Gradient anchor tensors tracked per device.
        cur_device: Index of the device used in the latest scheduling step.
        n_devices: Total number of CUDA devices detected.
    """

    def __init__(self):
        """Pre-allocate gradient anchor tensors per CUDA device."""
        self.tags: List[torch.Tensor] = [
            torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
            for _ in range(torch.cuda.device_count())
        ]
        self.cur_device: int = 0
        self.n_devices: int = torch.cuda.device_count()

    def get_next_tag(self) -> torch.Tensor:
        """Return the tag tensor assigned to the next device in rotation."""
        self.cur_device = (self.cur_device + 1) % self.n_devices
        tag = self.tags[self.cur_device]
        return tag

    def update_current_tag(self, new_tag: torch.Tensor) -> None:
        """Cache the tag produced by the most recent backward pass."""
        self.tags[self.cur_device] = new_tag

    def reset(self) -> None:
        """Reset rotation state so unrelated runs do not share graphs."""
        self.cur_device = 0
        self.tags = [
            torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
            for _ in range(torch.cuda.device_count())
        ]


backward_schedule_simulator: BackwardScheduleSimulator = BackwardScheduleSimulator()


def chunk_layer_params(
    tensor_pair: List[Tuple[torch.Tensor, torch.Tensor]], n_chunks: int
) -> List[List[Tuple[torch.Tensor, torch.Tensor]]]:
    """Group tensor copies into balanced chunks for overlapped transfers.

    Here we use a greedy number partitioning algorithm to distribute tensor
    copy work across multiple chunks. This helps balance the workload when
    transferring model parameters.

    Args:
        tensor_pair: List of ``(src, dst)`` tensors representing copy work.
        n_chunks: Number of chunks/events to distribute the work across.

    Returns:
        List of chunks where each chunk is a list of tensor copy pairs.
    """

    def get_tensor_size(pair: Tuple[torch.Tensor, torch.Tensor]) -> int:
        """Return the total bytes represented by ``pair``."""
        src, dst = pair
        return src.numel() * src.element_size()

    tensor_pair.sort(key=get_tensor_size, reverse=True)
    chunk_scheme: List[List[Tuple[torch.Tensor, torch.Tensor]]] = [
        [] for _ in range(n_chunks)
    ]
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
