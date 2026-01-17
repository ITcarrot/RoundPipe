"""Module defining parameter attributes for RoundPipe."""

from typing_extensions import *
import threading

import torch

from .threads import dump_all_active_threads, KeyboardInterruptRoundPipeThreads


class ParamAttribute:
    """Class storing parameter attributes for RoundPipe.

    Attributes:
        data_cpu: The CPU copy of the parameter data.
        data_grad: A temporary location to hold reference to gradient.
        data_optim: The optimizer copy of the parameter data.
        optim_grad: Hold reference to optimizer gradient to avoid reallocation.
        uploaded_grad: Whether the gradient is uploaded from cpu.
    """

    @classmethod
    def set(cls, t: torch.Tensor) -> None:
        """Attach a ParamAttribute to a tensor."""
        attr = cls(t.data)
        t.roundpipe_param_attr = attr  # pyright: ignore[reportAttributeAccessIssue]

    @classmethod
    def has(cls, t: torch.Tensor) -> bool:
        """Check if a tensor has a ParamAttribute attached."""
        return hasattr(t, "roundpipe_param_attr")

    @classmethod
    def get(cls, t: torch.Tensor) -> "ParamAttribute":
        """Get the ParamAttribute attached to a tensor."""
        return t.roundpipe_param_attr  # pyright: ignore[reportAttributeAccessIssue]

    def __init__(self, data: torch.Tensor) -> None:
        """Initialize the ParamAttribute with the given data.

        Args:
            data: The `.data` object of the parameter tensor.
        """
        self.data_cpu: torch.Tensor = data
        self.data_grad: Optional[torch.Tensor] = None
        self.data_optim: torch.Tensor = data
        self.optim_grad: Optional[torch.Tensor] = None
        self.uploaded_grad: bool = False

    def optim_inited(self) -> bool:
        """Check if the optimizer copy has been initialized."""
        return self.data_optim is not self.data_cpu


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
        except KeyboardInterrupt:
            dump_all_active_threads()
            raise KeyboardInterruptRoundPipeThreads from None
        if clear:
            self.param_upload_started.clear()

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
            self.grad_copied.wait()
            self.grad_download_started.wait()
            self.grad_download_started.clear()
            self.grad_downloaded.synchronize()
        except KeyboardInterrupt:
            dump_all_active_threads()
            raise KeyboardInterruptRoundPipeThreads from None
