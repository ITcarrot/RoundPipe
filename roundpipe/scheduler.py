"""Scheduling utilities that orchestrate forward/backward ordering.

Attributes:
    backward_schedule_simulator: Global simulator for pipelined backward scheduling.
"""

from typing_extensions import *
import threading
import heapq
import copy

import torch

from .threads import dump_all_active_threads, KeyboardInterruptRoundPipeThreads

if TYPE_CHECKING:
    from .roundpipe import RoundPipe
else:
    RoundPipe = TypeAliasType("RoundPipe", "roundpipe.roundpipe.RoundPipe")


class ModelExecutePlan:
    """Execution plans for a RoundPipe model.

    Attributes:
        fwd_plan: List of layers execution orders during forward.
        bwd_plan: List of layers execution orders during backward.
    """

    def __init__(self) -> None:
        """Initialize empty execution plans."""
        self.fwd_plan: List[range] = []
        self.bwd_plan: List[range] = []

    def __repr__(self) -> str:
        """Return string representation of the execution plans."""
        return f"Fwd Plan: {self.fwd_plan}, Bwd Plan: {self.bwd_plan}"

    def check_valid(
        self, num_layers: int, run_type: Literal["infer", "train", "fused"]
    ) -> None:
        """Validate that the execution plans cover all layers exactly once.

        Args:
            num_layers: Total number of layers in the model.
            run_type: Type of model run.

        Raises:
            ValueError: If the plans do not cover all layers exactly once.
        """
        cur_fwd_layer = -1
        for layer_range in self.fwd_plan:
            if len(layer_range) == 0:
                raise ValueError("Empty layer range in forward plan")
            for layer_id in layer_range:
                if layer_id != cur_fwd_layer + 1:
                    raise ValueError(
                        f"Specify {layer_id} after {cur_fwd_layer} in forward plan"
                    )
                cur_fwd_layer = layer_id
        if run_type in ("infer", "train"):
            if cur_fwd_layer != num_layers - 1:
                raise ValueError(
                    f"Forward plan does not cover all layers, ending at {cur_fwd_layer}"
                )
        if run_type == "infer":
            return
        cur_bwd_layer = -1
        for layer_range in reversed(self.bwd_plan):
            if len(layer_range) == 0:
                raise ValueError("Empty layer range in backward plan")
            for layer_id in layer_range:
                if layer_id != cur_bwd_layer + 1:
                    raise ValueError(
                        f"Specify {layer_id} before {cur_bwd_layer} in backward plan"
                    )
                cur_bwd_layer = layer_id
        if run_type == "train" and cur_bwd_layer != num_layers - 1:
            raise ValueError(
                f"Backward plan does not cover all layers, ending at {cur_bwd_layer}"
            )
        if run_type == "fused" and cur_fwd_layer + 1 != self.bwd_plan[0][0]:
            raise ValueError(
                f"Fused plan does not cover all layers, "
                f"mismatch forward between {cur_fwd_layer} and {self.bwd_plan[0][0]}"
            )

    @overload
    @classmethod
    def auto(
        cls,
        run_type: Literal["infer", "train", "fused"],
        model: "RoundPipe",
        /,
        *,
        min_stages: int = torch.cuda.device_count(),
        upper_threshold: float = 1.5,
    ) -> "ModelExecutePlan": ...
    @overload
    @classmethod
    def auto(
        cls,
        run_type: Literal["infer", "train", "fused"],
        model1: "RoundPipe",
        model2: "RoundPipe",
        /,
        *models: "RoundPipe",
        min_stages: int = torch.cuda.device_count(),
        upper_threshold: float = 1.5,
    ) -> List["ModelExecutePlan"]: ...
    @classmethod
    def auto(
        cls,
        run_type: Literal["infer", "train", "fused"],
        /,
        *models: "RoundPipe",
        min_stages: int = torch.cuda.device_count(),
        upper_threshold: float = 1.1,
    ) -> Union["ModelExecutePlan", List["ModelExecutePlan"]]:
        """Generate automatic execution plans based on model timings.

        Args:
            run_type: Type of model run.
            models: One or more ``RoundPipe`` models to base the plans on.
            min_stages: Minimum number of pipeline stages to use. This is
                a hint for the planner, and the actual number of stages could
                be lower depending on the model size.
            upper_threshold: Upper threshold for stage balancing. This limits
                the maximum allowed ratio between stages and the slowest layer.
                Increasing this value provides more flexibility in balancing
                stages at the cost of consuming more GPU memory.

        Returns:
            List of ``ModelExecutePlan`` instances, one per model.
        """
        if len(models) == 0:
            return []
        n_models = len(models)

        workloads: List[Tuple[List[float], List[float]]] = []
        max_layer_workload = 0.0
        for model in models:
            src, fwd_times, bwd_times = model.model_timer.get_estimate(run_type)
            if src == "memory" and len(models) > 1:
                return [
                    cls.auto(
                        run_type,
                        model,
                        min_stages=min_stages,
                        upper_threshold=upper_threshold,
                    )
                    for model in models
                ]
            max_layer_workload = max(max_layer_workload, *fwd_times, *bwd_times)
            workloads.append((fwd_times, bwd_times))

        stage_workload_candidates: List[float] = []
        for model_times in workloads:
            for run_times in model_times:
                for start in range(len(run_times)):
                    prefix_sum = 0.0
                    for end in range(start, len(run_times)):
                        prefix_sum += run_times[end]
                        if prefix_sum > max_layer_workload * upper_threshold:
                            break
                        if prefix_sum >= max_layer_workload:
                            stage_workload_candidates.append(prefix_sum)
        stage_workload_candidates.sort()

        min_cost = float("inf")
        best_plan: List[ModelExecutePlan] = []
        for max_stage_workload in stage_workload_candidates:
            total_stages = 0
            cur_plans: List[ModelExecutePlan] = []
            for idx, (fwd_times, bwd_times) in enumerate(workloads):
                plan = ModelExecutePlan()
                if run_type == "infer":
                    reversed_plan: List[List[int]] = []
                    stage_sum = float("inf")
                    for layer_id in range(len(fwd_times) - 1, -1, -1):
                        if stage_sum + fwd_times[layer_id] > max_stage_workload:
                            reversed_plan.append([])
                            stage_sum = 0.0
                        reversed_plan[-1].append(layer_id)
                        stage_sum += fwd_times[layer_id]
                    for reversed_stage in reversed(reversed_plan):
                        plan.fwd_plan.append(
                            range(reversed_stage[-1], reversed_stage[0] + 1)
                        )
                else:
                    layer_end = len(fwd_times)
                    if run_type == "fused" and idx == n_models - 1:
                        fused_stage_sum = 0.0
                        for layer_id in range(len(bwd_times) - 1, -1, -1):
                            if (
                                fused_stage_sum + bwd_times[layer_id]
                                > max_stage_workload
                            ):
                                break
                            fused_stage_sum += bwd_times[layer_id]
                            layer_end = layer_id

                    reversed_plan: List[List[int]] = []
                    stage_sum = float("inf")
                    for layer_id in range(layer_end - 1, -1, -1):
                        if stage_sum + fwd_times[layer_id] > max_stage_workload:
                            reversed_plan.append([])
                            stage_sum = 0.0
                        reversed_plan[-1].append(layer_id)
                        stage_sum += fwd_times[layer_id]
                    for reversed_stage in reversed(reversed_plan):
                        plan.fwd_plan.append(
                            range(reversed_stage[-1], reversed_stage[0] + 1)
                        )

                    reversed_plan = []
                    stage_sum = float("inf")
                    for layer_id in range(layer_end):
                        if stage_sum + bwd_times[layer_id] > max_stage_workload:
                            reversed_plan.append([])
                            stage_sum = 0.0
                        reversed_plan[-1].append(layer_id)
                        stage_sum += bwd_times[layer_id]
                    if run_type == "fused" and idx == n_models - 1:
                        plan.bwd_plan.append(range(layer_end, len(bwd_times)))
                    for stage in reversed(reversed_plan):
                        plan.bwd_plan.append(range(stage[0], stage[-1] + 1))

                cur_plans.append(plan)
                total_stages += len(plan.fwd_plan) + len(plan.bwd_plan)
            cur_cost = max(total_stages, min_stages) * max_stage_workload
            if cur_cost < min_cost:
                min_cost = cur_cost
                best_plan = cur_plans

        for idx, (model, plan) in enumerate(zip(models, best_plan)):
            actual_type = run_type
            if actual_type == "fused" and idx != n_models - 1:
                actual_type = "train"
            plan.check_valid(model.num_layers, actual_type)
        return best_plan[0] if len(best_plan) == 1 else best_plan


class ModelTracker:
    """Tracks forward and backward execution plans for a RoundPipe model.
    Contains semaphores to coordinate layer groups execution.

    Attributes:
        fwd_plan: List of layer ranges executed during forward.
        bwd_plan: List of layer ranges executed during backward.
        fwd_sem: Per-layer semaphores used to gate forward progress.
        bwd_sem: Per-layer semaphores used to gate backward progress.
    """

    def __init__(self, execute_plan: ModelExecutePlan) -> None:
        """Initialize ModelTracker based on the model configuration.

        Args:
            execute_plan: ``ModelExecutePlan`` instance with execution plans.
        """
        self.fwd_plan: List[range] = copy.deepcopy(execute_plan.fwd_plan)
        self.bwd_plan: List[range] = copy.deepcopy(execute_plan.bwd_plan)
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
