"""A gradient scaler for mixed precision training in RoundPipe."""

from typing_extensions import *

import torch
from torch.optim.optimizer import Optimizer

from .optim_stream import on_optim_stream, launch_optim_kernel, synchronize_optim
from .threads import AnnotatedEvent


class GradScaler:
    """Helps perform the steps of gradient scaling conveniently.
    The GradScaler is designed to be API compatible with `torch.amp.GradScaler`.

    !!! info
        Object of this class will be access both from the main thread
        and from the optimizer stream. Care must be taken to avoid data
        corruption due to race conditions. All methods in this class
        must check which stream they are called from, and behave accordingly.
        The following object members access rules are applied:

        * Both: `next_scale`, `scaler_updated`
        * Optimizer stream only: `main_scaler`
        * Main thread only: `scale_scaler`
        * Read only: `enabled`

    Attributes:
        enabled: Whether gradient scaling is enabled.
        main_scaler: The main GradScaler tracks the scale used for unscaling and updating.
        scale_scaler: This GradScaler is only used for applying scaling to outputs.
        next_scale: The next scale factor to be used.
        scaler_updated: An event to signal when the scaler has been updated.
    """

    def __init__(
        self,
        init_scale: float = 2.0**16,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        growth_interval: int = 2000,
        enabled: bool = True,
    ) -> None:
        """
        Args:
            init_scale: Initial scale factor.
            growth_factor: Factor by which the scale is multiplied during
                `update` if no inf/NaN gradients occur for ``growth_interval`` consecutive iterations.
            backoff_factor: Factor by which the scale is multiplied during
                `update` if inf/NaN gradients occur in an iteration.
            growth_interval: Number of consecutive iterations without inf/NaN gradients
                that must occur for the scale to be multiplied by ``growth_factor``.
            enabled: If ``False``, disables gradient scaling. `step` simply
                invokes the underlying ``optimizer.step()``, and other methods become no-ops.
        """
        self.enabled: Final[bool] = enabled
        self.main_scaler: torch.GradScaler = torch.GradScaler(
            "cpu", init_scale, growth_factor, backoff_factor, growth_interval
        )
        self.scale_scaler: torch.GradScaler = torch.GradScaler("cpu", init_scale)
        self.next_scale: torch.Tensor = torch.full(
            (), init_scale, dtype=torch.float32, device="cpu"
        )
        self.scaler_updated: AnnotatedEvent = AnnotatedEvent(f"Gscal_upd")
        self.scaler_updated.set()

        self.main_scaler._lazy_init_scale_growth_tracker(torch.device("cpu"))
        self.scale_scaler._lazy_init_scale_growth_tracker(torch.device("cpu"))

    def _launch_kernel(
        self, fn: Callable, sync: bool, *args: Any, **kwargs: Any
    ) -> Optional[Any]:
        """Make sure the fn executes on the optimizer stream.

        Args:
            fn: Function to launch.
            sync: Whether to synchronize after launching the kernel.
            args: Arguments to the function.
            kwargs: Keyword arguments to the function.

        Returns:
            If we are on the optim stream, returns the return value of fn.
                Otherwise, returns None.
        """
        if on_optim_stream():
            return fn(*args, **kwargs)
        else:
            launch_optim_kernel(fn, *args, **kwargs)
            if sync:
                synchronize_optim()

    @overload
    def scale(self, outputs: torch.Tensor) -> torch.Tensor: ...
    @overload
    def scale(self, outputs: List[torch.Tensor]) -> List[torch.Tensor]: ...
    @overload
    def scale(self, outputs: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]: ...
    @overload
    def scale(self, outputs: Iterable[torch.Tensor]) -> Iterable[torch.Tensor]: ...
    def scale(
        self,
        outputs: Union[torch.Tensor, Iterable[torch.Tensor]],
    ) -> Union[torch.Tensor, Iterable[torch.Tensor]]:
        """
        Multiplies ('scales') a tensor or list of tensors by the scale factor.

        Returns scaled outputs. If this instance of `GradScaler` is not enabled, outputs are returned
        unmodified.

        Args:
            outputs: Outputs to scale.
        """
        if not self.enabled:
            return outputs
        if on_optim_stream():
            return self.main_scaler.scale(outputs)
        else:
            return self.scale_scaler.scale(outputs)

    def unscale_(self, optimizer: Optimizer) -> None:
        """
        Divides ("unscales") the optimizer's gradient tensors by the scale factor.
        `unscale_` is optional, serving cases where you need to modify or inspect
        gradients between the backward pass(es) and `step`. If `unscale_` is not
        called explicitly, gradients will be unscaled automatically during `step`.

        If this is called from the main thread, the unscale operation will be
        launched and synchronized on the optimizer stream.

        Args:
            optimizer: Optimizer that owns the gradients to be unscaled.
        """
        if not self.enabled:
            return
        self._launch_kernel(self.main_scaler.unscale_, True, optimizer)

    def step(self, optimizer: Optimizer, *args: Any, **kwargs: Any) -> Optional[float]:
        """Invoke ``unscale_(optimizer)`` followed by parameter update, if gradients are not infs/NaN.

        `step` carries out the following two operations:

        1.  Internally invokes ``unscale_(optimizer)`` (unless `unscale_` was explicitly called for ``optimizer``
            earlier in the iteration).  As part of the `unscale_`, gradients are checked for infs/NaNs.
        2.  If no inf/NaN gradients are found, invokes ``optimizer.step()`` using the unscaled
            gradients.  Otherwise, ``optimizer.step()`` is skipped to avoid corrupting the params.

        ``*args`` and ``**kwargs`` are forwarded to ``optimizer.step()``.

        Args:
            optimizer:  Optimizer that applies the gradients.
            args:  Any arguments.
            kwargs:  Any keyword arguments.

        Returns:
            If it's disabled, returns the return value of ``optimizer.step(*args, **kwargs)``.
                If enabled, it returns the value only if we are on the optim stream.
        """
        if not self.enabled:
            return optimizer.step(*args, **kwargs)
        return self._launch_kernel(
            self.main_scaler.step, True, optimizer, *args, **kwargs
        )

    def update_kernel(self, new_scale: Optional[Union[float, torch.Tensor]]) -> None:
        """Kernel function to update the scale factor on the optimizer stream.

        Args:
            new_scale:  New scale factor.
        """
        # Always update the newest scale
        assert self.main_scaler._scale is not None
        self.main_scaler._scale.copy_(self.next_scale)
        self.main_scaler.update(new_scale)
        # A new scale is generated, but not used until next update() call
        # at main thread. Do a swap here.
        tmp = self.next_scale.clone()
        self.next_scale.copy_(self.main_scaler._scale)
        self.main_scaler._scale.copy_(tmp)
        self.scaler_updated.set()

    def update(self, new_scale: Optional[Union[float, torch.Tensor]] = None, is_async: bool = True) -> None:
        """Update the scale factor. This function must be called from the main thread.

        If any optimizer steps were skipped the scale is multiplied by ``backoff_factor``
        to reduce it. If ``growth_interval`` unskipped iterations occurred consecutively,
        the scale is multiplied by ``growth_factor`` to increase it.

        Passing ``new_scale`` sets the new scale value manually. (``new_scale`` is not
        used directly, it's used to fill GradScaler's internal scale tensor. So if
        ``new_scale`` was a tensor, later in-place changes to that tensor will not further
        affect the scale GradScaler uses internally.)

        Args:
            new_scale:  New scale factor.
            is_async: If True, the update will be performed asynchronously on the optimizer stream.
                If False, this function will block until the update is complete. Note that if
                ``is_async=False``, the main thread will wait for the optimizer stream to finish
                processing all previously launched kernels, including the optimizer step, which may
                cause significant slowdown.
        """
        if not self.enabled:
            return
        if on_optim_stream():
            raise RuntimeError(
                "GradScaler.update() must be called from the main thread."
            )

        self.scaler_updated.wait()
        assert self.scale_scaler._scale is not None
        self.scale_scaler._scale.copy_(self.next_scale)
        self.scaler_updated.clear()
        launch_optim_kernel(self.update_kernel, new_scale)
        if not is_async:
            self.scaler_updated.wait()
            self.scale_scaler._scale.copy_(self.next_scale)
            assert self.main_scaler._scale is not None
            self.main_scaler._scale.copy_(self.next_scale)

    def get_scale(self) -> float:
        """Return the current scale factor. Result will adapt to which stream this is called from.

        Returns:
            a Python float containing the current scale, or 1.0 if scaling is disabled.
        """
        if on_optim_stream():
            return self.main_scaler.get_scale() if self.enabled else 1.0
        else:
            return self.scale_scaler.get_scale() if self.enabled else 1.0

    def get_growth_factor(self, up_to_date: bool = False) -> float:
        """
        Args:
            up_to_date:  If True, make sure to return the latest growth factor,
                but will block and synchronize with the optimizer stream. Else, may return
                a stale value not before the previous ``GradScaler.update()``.
        Returns:
            a Python float containing the scale growth factor.
        """
        if up_to_date and not on_optim_stream():
            synchronize_optim()
        return self.main_scaler.get_growth_factor()

    def set_growth_factor(self, new_factor: float) -> None:
        """Set a new scale growth factor.

        Args:
            new_factor:  Value to use as the new scale growth factor.
        """
        self._launch_kernel(self.main_scaler.set_growth_factor, False, new_factor)

    def get_backoff_factor(self, up_to_date: bool = False) -> float:
        """
        Args:
            up_to_date:  If True, make sure to return the latest backoff factor,
                but will block and synchronize with the optimizer stream. Else, may return
                a stale value not before the previous ``GradScaler.update()``.

        Returns:
            a Python float containing the scale backoff factor.
        """
        if up_to_date and not on_optim_stream():
            synchronize_optim()
        return self.main_scaler.get_backoff_factor()

    def set_backoff_factor(self, new_factor: float) -> None:
        """Set a new scale backoff factor.

        Args:
            new_factor:  Value to use as the new scale backoff factor.
        """
        self._launch_kernel(self.main_scaler.set_backoff_factor, False, new_factor)

    def get_growth_interval(self, up_to_date: bool = False) -> int:
        """
        Args:
            up_to_date:  If True, make sure to return the latest growth interval,
                but will block and synchronize with the optimizer stream. Else, may return
                a stale value not before the previous ``GradScaler.update()``.

        Returns:
            a Python int containing the growth interval.
        """
        if up_to_date and not on_optim_stream():
            synchronize_optim()
        return self.main_scaler.get_growth_interval()

    def set_growth_interval(self, new_interval: int) -> None:
        """Set a new growth interval.

        Args:
            new_interval: Value to use as the new growth interval.
        """
        self._launch_kernel(self.main_scaler.set_growth_interval, False, new_interval)
