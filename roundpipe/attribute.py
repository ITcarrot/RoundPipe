"""Module defining parameter attributes for RoundPipe."""

from typing_extensions import *
import threading

import torch

from .threads import dump_all_active_threads, KeyboardInterruptRoundPipeThreads


class ParamAttribute:
    """Class storing parameter attributes for RoundPipe.
    Note that parameters may be shared among multiple layers.

    Attributes:
        grad_cpu: Dictionary mapping layer IDs to gradient tensors
            stored on CPU.
        grad_buffer: Dictionary mapping layer IDs to a temporary location
            that hold reference to the respective gradient. This is used
            to avoid memory re-allocation between gradient downloads.
        optim: Tensor storing a copy of the parameter for optimizer use.
        optim_grad_buffer: Hold reference to optimizer gradient to avoid reallocation.
    """

    def __init__(self) -> None:
        """Initialize the ParamAttribute with the given data."""
        self.grad_cpu: Dict[int, Optional[torch.Tensor]] = {}
        self.grad_buffer: Dict[int, Optional[torch.Tensor]] = {}
        self.optim: Optional[torch.nn.Parameter] = None
        self.optim_grad_buffer: Optional[torch.Tensor] = None

    @classmethod
    def set(cls, t: torch.nn.Parameter, layer_id: Optional[int]) -> None:
        """Attach a ParamAttribute to a tensor.

        Args:
            t: Tensor to attach the attribute to.
            layer_id: Object ID of the layer this parameter belongs to.
        """
        attr: ParamAttribute = getattr(t, "roundpipe_param_attr", cls())
        if layer_id is not None:
            attr.grad_cpu[layer_id] = None
            attr.grad_buffer[layer_id] = None
        t.roundpipe_param_attr = attr  # pyright: ignore[reportAttributeAccessIssue]

    @classmethod
    def has(cls, t: torch.nn.Parameter) -> bool:
        """Check if a tensor has a ParamAttribute attached."""
        return hasattr(t, "roundpipe_param_attr")

    @classmethod
    def get(cls, t: torch.nn.Parameter) -> "ParamAttribute":
        """Get the ParamAttribute attached to a tensor."""
        return t.roundpipe_param_attr  # pyright: ignore[reportAttributeAccessIssue]


class LayerAttribute:
    """Class storing layer attributes for RoundPipe. Events
    are used to synchronize layer param and grad between device,
    host, and optimizer threads.

    Event flow (W&C = wait and clear):
    {wait param_copied -> W&C param_upload_started ->
    launch forward thread[-> set param_upload_started]} ->
    {wait grad_copied -> W&C grad_download_started ->
    launch backward thread[-> set grad_download_started]} ->
    {W&C param_copied -> W&C grad_copied ->
    launch optimizer thread[wait param_upload_started -> do copy ->
    set param_copied -> wait grad_download_started -> do copy ->
    set grad_copied]} -> repeat

    Attributes:
        param_copied: Event indicating parameter copy from optimizer to
            model is done.
        param_upload_started: Event indicating parameter upload from CPU
            to GPU has started and CUDA event has been recorded.
        param_uploaded: CUDA event indicating parameter upload to device is done.
        grad_copied: Event indicating gradient copy from model to optimizer
            is done.
        grad_download_started: Event indicating gradient download from GPU
            to CPU has started and CUDA event has been recorded.
        grad_downloaded: CUDA event indicating gradient download to CPU is done.
    """

    def __init__(self) -> None:
        self.param_copied: threading.Event = threading.Event()
        self.param_copied.set()
        self.param_upload_started: threading.Event = threading.Event()
        self.param_upload_started.set()
        self.param_uploaded: torch.cuda.Event = cast(
            torch.cuda.Event, torch.cuda.Event()
        )

        self.buffer_download_started: threading.Event = threading.Event()
        self.buffer_download_started.set()
        self.buffer_downloaded: torch.cuda.Event = cast(
            torch.cuda.Event, torch.cuda.Event()
        )

        self.grad_copied: threading.Event = threading.Event()
        self.grad_copied.set()
        self.grad_download_started: threading.Event = threading.Event()
        self.grad_download_started.set()
        self.grad_downloaded: torch.cuda.Event = cast(
            torch.cuda.Event, torch.cuda.Event()
        )

    def forward_fence(self, clear: bool = True) -> None:
        """Fence for forward pass to wait for parameter ready.

        Args:
            clear: Whether to clear events for doing forward.
        """
        try:
            self.param_copied.wait()
            self.param_upload_started.wait()
            self.buffer_download_started.wait()
            self.buffer_downloaded.synchronize()
        except KeyboardInterrupt:
            dump_all_active_threads()
            raise KeyboardInterruptRoundPipeThreads from None
        if clear:
            self.param_upload_started.clear()
            self.buffer_download_started.clear()

    def backward_fence(self, clear: bool = True) -> None:
        """Fence for backward pass to wait for parameter and gradient ready.

        Args:
            clear: Whether to clear events for doing backward.
        """
        try:
            self.param_copied.wait()
            self.param_upload_started.wait()
            self.grad_copied.wait()
            self.grad_download_started.wait()
            self.grad_downloaded.synchronize()
        except KeyboardInterrupt:
            dump_all_active_threads()
            raise KeyboardInterruptRoundPipeThreads from None
        if clear:
            self.param_upload_started.clear()
            self.grad_download_started.clear()

    def forward_backward_fence(self) -> None:
        """Fence for fused forward backward pass to wait for
        parameter and gradient ready.

        Args:
            clear: Whether to clear events for doing backward.
        """
        try:
            self.param_copied.wait()
            self.param_upload_started.wait()
            self.param_upload_started.clear()
            self.buffer_download_started.wait()
            self.buffer_download_started.clear()
            self.buffer_downloaded.synchronize()
            self.grad_copied.wait()
            self.grad_download_started.wait()
            self.grad_download_started.clear()
            self.grad_downloaded.synchronize()
        except KeyboardInterrupt:
            dump_all_active_threads()
            raise KeyboardInterruptRoundPipeThreads from None
