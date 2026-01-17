"""Managing each CUDA device on dedicated threads.

Attributes:
    device_list: List of instantiated ``DeviceManager`` objects.
    cur_device: Index tracking the next device to schedule work on.
"""

from typing_extensions import *
import threading
import itertools

import torch

from .attribute import LayerAttribute
from .run import run_forward, run_backward, run_forward_backward, RoundPipeRunContext
from .threads import (
    RoundPipeThread,
    dump_all_active_threads,
    KeyboardInterruptRoundPipeThreads,
)

if TYPE_CHECKING:
    from .batch import Batch
else:
    Batch = TypeAliasType("Batch", "roundpipe.batch.Batch")


class InterStreamMemManager:
    """Handles tensor deallocation across multiple CUDA streams.

    Attributes:
        free_queue: Maps stream pairs to lists of tensor storage
            waiting to be freed after synchronization.
        in_use: Maps tensor storages to the list of streams that
            have used them.
    """

    def __init__(self, *streams: torch.cuda.Stream):
        """Initialize tracking structures for inter-stream memory management.

        Args:
            streams: Streams that will be used by tensors.
        """
        self.free_queue: Dict[
            Tuple[torch.cuda.Stream, torch.cuda.Stream],
            List[Tuple[torch.UntypedStorage, Tuple[torch.cuda.Stream, ...]]],
        ] = {}
        self.in_use: Dict[torch.UntypedStorage, List[torch.cuda.Stream]] = {}
        for src, dst in itertools.product(streams, streams):
            if src is not dst:
                self.free_queue[(src, dst)] = []

    def record_stream(
        self,
        tensor: torch.Tensor,
        from_stream: torch.cuda.Stream,
        use_stream: torch.cuda.Stream,
    ) -> None:
        """Record that ``tensor`` is used on ``use_stream`` after being
        used on ``from_stream``.

        Args:
            tensor: Tensor whose usage is being tracked.
            from_stream: Stream where the tensor was last used.
            use_stream: Stream where the tensor will be used next.

        Raises:
            AssertionError: If the last recorded stream for ``tensor``
                does not match ``from_stream``.
        """
        storage = tensor.untyped_storage()
        if storage not in self.in_use:
            self.in_use[storage] = [from_stream, use_stream]
        elif from_stream in self.in_use[storage] and use_stream in self.in_use[storage]:
            return
        else:
            assert (
                self.in_use[storage][-1] is from_stream
            ), "Tensor usage tracking: expected last stream to match from_stream."
            self.in_use[storage].append(use_stream)

    def stream_synced(
        self, waiter: torch.cuda.Stream, wait_for: torch.cuda.Stream
    ) -> None:
        """When ``waiter`` stream has synchronized on ``wait_for``,
        release all tensors waiting on this synchronization.

        Args:
            waiter: Stream that has synchronized.
            wait_for: Stream that ``waiter`` has synchronized on.
        """
        for storage, use_streams in self.free_queue[(waiter, wait_for)]:
            if len(use_streams) > 2:
                self.free_queue[(use_streams[-3], use_streams[-2])].append(
                    (storage, use_streams[:-1])
                )
        self.free_queue[(waiter, wait_for)].clear()

    def free(
        self, storage: torch.UntypedStorage, *use_streams: torch.cuda.Stream
    ) -> None:
        """Hold tensor ``storage`` reference before all streams
        used this storage synchronize back to the owning stream.

        Args:
            storage: Tensor storage to be freed.
            use_streams: Ordered streams that have used ``storage``.
        """
        if len(use_streams) < 2:
            return
        self.free_queue[(use_streams[-2], use_streams[-1])].append(
            (storage, use_streams)
        )

    def flush(self) -> None:
        """Flush all tracked tensors into free queues."""
        for storage, use_streams in self.in_use.items():
            self.free(storage, *use_streams)
        self.in_use = {}

    def free_all(self) -> None:
        """Clear all tracked tensors from free queues"""
        for _, track_list in self.free_queue.items():
            for storage, streams in track_list:
                tensor = torch.empty(0, device=storage.device).set_(storage)
                for stream in streams:
                    tensor.record_stream(stream)
        self.free_queue = {k: [] for k in self.free_queue}


class DeviceManager:
    """Manages memory, CUDA streams and kernel launch of a single device.
    !!! info
        The device controller threads is designed to run multiple forward
        / backward jobs concurrently on different devices. It is NOT designed
        to run jobs while doing other work on the main thread at the same time.
        The main thread should wait for all device threads to return before
        proceeding to the next step.

    Attributes:
        id: Integer identifier matching ``cuda:{id}``.
        device: Corresponding Pytorch CUDA device handle.
        param_upstream: Stream for model parameter uploads.
        upstream: Stream handling activation uploads.
        compute_stream: Default compute stream bound to ``device``.
        downstream: Stream used for downloads to host.
        mem_manager: Inter-stream memory manager for this device.
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

        self.param_upstream: torch.cuda.Stream = cast(
            torch.cuda.Stream, torch.cuda.Stream(device)
        )
        self.upstream: torch.cuda.Stream = cast(
            torch.cuda.Stream, torch.cuda.Stream(device)
        )
        self.compute_stream: torch.cuda.Stream = torch.cuda.default_stream(self.device)
        self.downstream: torch.cuda.Stream = cast(
            torch.cuda.Stream, torch.cuda.Stream(device)
        )
        self.mem_manager: InterStreamMemManager = InterStreamMemManager(
            self.param_upstream, self.upstream, self.compute_stream, self.downstream
        )
        self.upload_mark: List[torch.cuda.Event] = []

        self.is_idle: threading.Semaphore = threading.Semaphore(1)
        self.job_arrived: threading.Semaphore = threading.Semaphore(0)
        self.cur_job: Optional[
            Tuple[Callable[..., None], List[RoundPipeRunContext], Tuple]
        ] = None
        self.controller_thread: RoundPipeThread = RoundPipeThread(
            target=self.controller, name=f"RoundPipe DeviceController-{id}"
        )

    def wait_stream(
        self, waiter: torch.cuda.Stream, wait_for: torch.cuda.Stream
    ) -> None:
        """Make ``waiter`` stream wait on ``wait_for`` stream.

        Args:
            waiter: Stream that will wait.
            wait_for: Stream to wait on.
        """
        waiter.wait_stream(wait_for)
        self.mem_manager.stream_synced(waiter, wait_for)

    def mark_upload(self) -> None:
        """Record an event on the upload stream after enqueuing H2D copies.

        Returns:
            The recorded event is appended to ``upload_mark``.
        """
        event = cast(torch.cuda.Event, torch.cuda.Event())
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
            fn, run_context, args = self.cur_job

            for batch_context in run_context:
                fn(self, batch_context, *args)

            # clear variables references to avoid memory waste
            self.cur_job = None
            del fn, run_context, args
            self.controller_thread.is_active = False
            self.is_idle.release()

    def launch_forward(
        self,
        layer_group_id: int,
        layer_attrs: List[LayerAttribute],
        batch: "Batch",
        run_context: List[RoundPipeRunContext],
    ) -> None:
        """Schedule a forward-only job on this device.

        Args:
            layer_group_id: Index into the execute plan's forward ordering.
            layer_attrs: List of layer attributes for the layers in this group.
            batch: Batch container shared across microbatches.
            run_context: Per-microbatch execution contexts.
        """
        try:
            self.is_idle.acquire()
        except KeyboardInterrupt:
            dump_all_active_threads()
            raise KeyboardInterruptRoundPipeThreads from None
        for layer_attr in layer_attrs:
            layer_attr.forward_fence()
        self.cur_job = (run_forward, run_context, (layer_group_id, batch))
        self.job_arrived.release()

    def launch_backward(
        self,
        layer_group_id: int,
        layer_attrs: List[LayerAttribute],
        run_context: List[RoundPipeRunContext],
    ) -> None:
        """Schedule a backward-only job on this device.

        Args:
            layer_group_id: Index into the execute plan's backward ordering.
            layer_attrs: List of layer attributes for the layers in this group.
            run_context: Per-microbatch execution contexts.
        """
        try:
            self.is_idle.acquire()
        except KeyboardInterrupt:
            dump_all_active_threads()
            raise KeyboardInterruptRoundPipeThreads from None
        for layer_attr in layer_attrs:
            layer_attr.backward_fence()
        self.cur_job = run_backward, run_context, (layer_group_id,)
        self.job_arrived.release()

    def launch_forward_backward(
        self,
        layer_attrs: List[LayerAttribute],
        batch: "Batch",
        run_context: List[RoundPipeRunContext],
        loss_fn: Callable[[Any, Any], Union[Sequence[torch.Tensor], torch.Tensor]],
        return_outputs: bool,
    ) -> None:
        """Run forward + backward in the same pass for final layers.

        Args:
            batch: Batch object generated for the training iteration.
            run_context: Per-microbatch execution contexts.
            loss_fn: Callable that consumes outputs + labels and returns loss.
            return_outputs: Whether to return model outputs from forward pass.
        """
        try:
            self.is_idle.acquire()
        except KeyboardInterrupt:
            dump_all_active_threads()
            raise KeyboardInterruptRoundPipeThreads from None
        for layer_attr in layer_attrs:
            layer_attr.forward_backward_fence()
        self.cur_job = (
            run_forward_backward,
            run_context,
            (batch, loss_fn, return_outputs),
        )
        self.job_arrived.release()


device_list: List[DeviceManager] = []
cur_device: int = 0

for i in range(torch.cuda.device_count()):
    device = DeviceManager(i, torch.device(f"cuda:{i}"))
    device_list.append(device)


def get_num_devices() -> int:
    """Return the number of managed GPU devices.

    Returns:
        Number of instantiated ``DeviceManager`` objects.
    """
    return len(device_list)


def get_next_device() -> DeviceManager:
    """Round-robin iterator over instantiated ``DeviceManager`` objects.

    Returns:
        The next device manager to schedule work on.
    """
    global cur_device
    device = device_list[cur_device]
    cur_device = (cur_device + 1) % len(device_list)
    return device


def gc_collect() -> None:
    """Release all tracked tensors from inter-stream memory
    managers on all devices.
    """
    for device in device_list:
        device.mem_manager.flush()
        device.mem_manager.free_all()
