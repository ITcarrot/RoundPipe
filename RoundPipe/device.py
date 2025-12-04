"""Managing each CUDA device on dedicated threads.

Attributes:
    device_list: List of instantiated ``DeviceManager`` objects.
    cur_device: Index tracking the next device to schedule work on.
"""

from beartype.typing import * # type: ignore[reportWildcardImport]
from enum import Enum
import threading

import torch

from .batch import Batch
from .run import run_forward, run_backward, run_forward_backward, RoundPipeRunContext
from .threads import RoundPipeThread, dump_all_active_threads, KeyboardInterruptRoundPipeThreads

class JobType(Enum):
    """Execution type scheduled onto a ``DeviceManager``."""

    FORWARD = 1
    BACKWARD = 2
    FORWARD_BACKWARD = 3

class DeviceManager:
    """Owns CUDA streams and a controller thread for a single device.

    Attributes:
        id: Integer identifier matching ``cuda:{id}``.
        device: Corresponding Pytorch CUDA device handle.
        param_upstream: Stream for model parameter uploads.
        upstream: Stream handling activation uploads.
        compute_stream: Default compute stream bound to ``device``.
        downstream: Stream used for downloads to host.
        upload_mark: Outstanding events that track chunked transfers.
        is_idle: Semaphore indicating when the controller can accept work.
        job_arrived: Semaphore signaled when a new job is queued.
        cur_job: Tuple describing the pending job type plus payload.
        controller_thread: Background thread executing queued jobs.
    """

    def __init__(self, id: int, device: torch.device):
        """Initialize CUDA streams and spawn the controller thread.

        Args:
            id: Sequential identifier assigned when enumerating devices.
            device: Corresponding Pytorch CUDA device handle.
        """
        self.id: int = id
        self.device: torch.device = device
        
        self.param_upstream: torch.cuda.Stream = torch.cuda.Stream(device) # type: ignore[reportAttributeAccessIssue]
        self.upstream: torch.cuda.Stream = torch.cuda.Stream(device) # type: ignore[reportAttributeAccessIssue]
        self.compute_stream = torch.cuda.default_stream(self.device)
        self.downstream: torch.cuda.Stream = torch.cuda.Stream(device) # type: ignore[reportAttributeAccessIssue]
        self.upload_mark: List[torch.cuda.Event] = []

        self.is_idle: threading.Semaphore = threading.Semaphore(1)
        self.job_arrived: threading.Semaphore = threading.Semaphore(0)
        self.cur_job: Optional[Tuple[JobType, int, Optional[Batch], List[RoundPipeRunContext],
                               Optional[Callable[[Any, Any], Union[Sequence[torch.Tensor], torch.Tensor]]]]] = None
        self.controller_thread: RoundPipeThread = RoundPipeThread(target=self.controller, name=f'RoundPipe DeviceController-{id}')

    def mark_upload(self) -> None:
        """Record an event on the upload stream after enqueuing H2D copies.

        Returns:
            The recorded event is appended to ``upload_mark``.
        """
        event: torch.cuda.Event = torch.cuda.Event() # type: ignore[reportAttributeAccessIssue]
        with torch.cuda.stream(self.upstream):
            event.record()
        self.upload_mark.append(event)

    def flush_upload_marks(self) -> List[torch.cuda.Event]:
        """Return outstanding upload markers and clear the queue.

        Returns:
            List of still-pending upload events. The list captures gaps between
                activation uploads so parameter transfers can be staged safely.
        """
        marks = [mark for mark in self.upload_mark if not mark.query()]
        self.upload_mark = []
        return marks

    def controller(self) -> None:
        """Background loop that executes jobs as they arrive.

        The loop blocks on ``job_arrived`` and dispatches to the appropriate
        runtime helper based on ``job_type``.
        """
        while True:
            self.job_arrived.acquire()
            self.controller_thread.is_active = True
            assert self.cur_job is not None
            job_type, layer_group_id, batch, run_context, loss_fn = self.cur_job

            if job_type == JobType.FORWARD:
                assert batch is not None
                for batch_context in run_context:
                    run_forward(layer_group_id, batch, batch_context, self)
            elif job_type == JobType.BACKWARD:
                for batch_context in run_context:
                    run_backward(layer_group_id, batch_context, self)
            elif job_type == JobType.FORWARD_BACKWARD:
                assert batch is not None and loss_fn is not None
                for batch_context in run_context:
                    run_forward_backward(batch, batch_context, loss_fn, self)

            # clear variables references to avoid memory waste
            self.cur_job = None
            del job_type, layer_group_id, batch, run_context, loss_fn
            self.controller_thread.is_active = False
            self.is_idle.release()

    def launch_forward(self, layer_group_id: int, batch: Batch,
                       run_context: List[RoundPipeRunContext]) -> None:
        """Schedule a forward-only job on this device.

        Args:
            layer_group_id: Index into the execute plan's forward ordering.
            batch: Batch container shared across microbatches.
            run_context: Per-microbatch execution contexts.
        """
        try:
            self.is_idle.acquire()
        except KeyboardInterrupt:
            dump_all_active_threads()
            raise KeyboardInterruptRoundPipeThreads from None
        self.cur_job = (JobType.FORWARD, layer_group_id, batch, run_context, None)
        self.job_arrived.release()
    
    def launch_backward(self, layer_group_id: int,
                        run_context: List[RoundPipeRunContext]) -> None:
        """Schedule a backward-only job on this device.

        Args:
            layer_group_id: Index into the execute plan's backward ordering.
            run_context: Per-microbatch execution contexts.
        """
        try:
            self.is_idle.acquire()
        except KeyboardInterrupt:
            dump_all_active_threads()
            raise KeyboardInterruptRoundPipeThreads from None
        self.cur_job = (JobType.BACKWARD, layer_group_id, None, run_context, None)
        self.job_arrived.release()

    def launch_forward_backward(self, batch: Batch, run_context: List[RoundPipeRunContext],
                                loss_fn: Callable[[Any, Any], Union[Sequence[torch.Tensor], torch.Tensor]]) -> None:
        """Run forward + backward in the same pass for final layers.

        Args:
            batch: Batch object generated for the training iteration.
            run_context: Per-microbatch execution contexts.
            loss_fn: Callable that consumes outputs + labels and returns loss.
        """
        try:
            self.is_idle.acquire()
        except KeyboardInterrupt:
            dump_all_active_threads()
            raise KeyboardInterruptRoundPipeThreads from None
        self.cur_job = (JobType.FORWARD_BACKWARD, 0, batch, run_context, loss_fn)
        self.job_arrived.release()

device_list: List[DeviceManager] = []
cur_device: int = 0

for i in range(torch.cuda.device_count()):
    device = DeviceManager(i, torch.device(f"cuda:{i}"))
    device_list.append(device)

def get_next_device() -> DeviceManager:
    """Round-robin iterator over instantiated ``DeviceManager`` objects.

    Returns:
        The next device manager to schedule work on.
    """
    global cur_device
    device = device_list[cur_device]
    cur_device = (cur_device + 1) % len(device_list)
    return device
