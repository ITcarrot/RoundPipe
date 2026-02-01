"""The RoundPipe model wrapper and execution runtime."""

from typing_extensions import *
import copy
import warnings

import tqdm
import torch
import torch.nn as nn

from .attribute import ParamAttribute, LayerAttribute
from .batch import Batch
from .context import doing_optimizer
from .device import get_next_device
from .optim_stream import launch_optim_kernel, on_optim_stream
from .run import (
    RoundPipeRunContext,
    RoundPipeBatchedBackward,
    RoundPipeMicrobatchBackward,
    RoundPipeInputBackward,
)
from .run_config import RoundPipeRunConfig, FullRoundPipeRunConfig
from .scheduler import ModelExecutePlan, ModelTracker, backward_schedule_simulator
from .threads import AnnotatedEvent
from .timer import ModelTimer, IterTimer
from .utils import get_call_location, pin_tensor, get_model_size


class RoundPipeBase(nn.Module):
    """Common attributes and methods of RoundPipe and AutoRoundPipe

    Attributes:
        name: Human-readable identifier shown in traces/logs.
        model: The provided module wrapped for RoundPipe execution.
        original_model: Reference to the pre-wrapped module when shimming
            attribute access.
        layer_attrs: List of ``LayerAttribute`` storing per-layer events.
        optim_dtype: Data type for optimizer parameters.
        optim_updated: Event signaling optimizer have updated.
    """

    def __init__(
        self,
        model: nn.Module,
        name: Optional[str] = None,
        optim_dtype: Optional[torch.dtype] = None,
    ) -> None:
        """Initialize the RoundPipe base wrapper.

        Args:
            model: Module to wrap.
            name: Optional friendly identifier. Defaults to ``file:line``.
            optim_dtype: Data type for optimizer parameters. Defaults to the same
                as the parameter data type.
        """
        super().__init__()
        # call stack: -> (Auto)RoundPipe -> RoundPipeBase
        self.name: str = name if name else get_call_location(2)
        self.model: nn.Module = model
        self.original_model: Optional[nn.Module] = (
            None  # placeholder for original model if needing its functions
        )

        self.layer_attrs: List[LayerAttribute] = []

        self.optim_dtype: Optional[torch.dtype] = optim_dtype
        self.optim_updated: AnnotatedEvent = AnnotatedEvent(f"{self.name}_opt")
        self.optim_updated.set()

    def __getattr__(self, name: str) -> Any:
        """Delegate missing attributes to the wrapped or original module."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            if self.original_model is not None:
                return getattr(self.original_model, name)
            return getattr(self.model, name)

    def __setattr__(self, name: str, value: Any) -> None:
        """Mirror attribute writes to wrapped/original models post-initialization."""
        if "RoundPipe_initialized" in self.__dict__:
            if self.original_model is not None:
                setattr(self.original_model, name, value)
            setattr(self.model, name, value)
        else:
            super().__setattr__(name, value)

    def __delattr__(self, name: str) -> None:
        """Ensure attribute deletions propagate to wrapped/original modules."""
        if "RoundPipe_initialized" in self.__dict__:
            if self.original_model is not None:
                delattr(self.original_model, name)
            delattr(self.model, name)
        else:
            return super().__delattr__(name)

    def set_original_model(self, original_model: nn.Module) -> None:
        """Attach the pre-wrap model for attribute shimming.

        Args:
            original_model: Module that should mirror attribute updates.
        """
        object.__setattr__(self, "original_model", original_model)

    @override
    def named_parameters(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ) -> Iterator[Tuple[str, nn.Parameter]]:
        """Iterator over named parameters. Overrides to warn against direct use,
        and redirect to optim_named_parameters under optimizer context.
        """
        if doing_optimizer() and recurse:
            return self.optim_named_parameters(prefix, remove_duplicate)
        warnings.warn(
            "RoundPipe will manage parameter location and dtype internally. "
            "\nAccessing parameters() or named_parameters() directly may "
            "lead to unexpected behavior. \nIf you intend to get parameters "
            "for optimization, please use optim_parameters() or "
            "optim_named_parameters() instead.",
            UserWarning,
        )
        return super().named_parameters(prefix, recurse, remove_duplicate)

    @override
    def parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]:
        """Iterator over parameters. Overrides to redirect to optim_parameters
        under optimizer context.
        """
        if doing_optimizer() and recurse:
            return self.optim_parameters()
        return super().parameters(recurse)

    def optim_named_parameters(
        self, prefix: str = "", remove_duplicate: bool = True
    ) -> Iterator[Tuple[str, torch.nn.Parameter]]:
        """Iterator over named parameters suitable for optimizer consumption.

        Args:
            prefix: Prefix to prepend to parameter names.
            remove_duplicate: Whether to skip duplicate parameters.

        Yields:
            Tuples of parameter names and their optimizer-ready tensors.
        """
        for name, param in super().named_parameters(prefix, True, remove_duplicate):
            param_attr = ParamAttribute.get(param)
            if param_attr.optim is None and param.requires_grad:
                param_attr.optim = torch.nn.Parameter(
                    param.to(dtype=self.optim_dtype, copy=True), param.requires_grad
                )
            if param_attr.optim is not None:
                yield name, param_attr.optim

    def optim_parameters(self) -> Iterator[torch.nn.Parameter]:
        """Iterator over parameters suitable for optimizer consumption.

        Yields:
            Parameters stored in their optimizer-ready format.
        """
        for _, param in self.optim_named_parameters():
            yield param

    def sync_optim_param(self) -> None:
        """Ensure optimizer updated results are copied back to parameters."""
        raise NotImplementedError("sync_optim_param must be implemented in subclasses.")

    def _move_grad_to_optim(self) -> None:
        """Move parameter gradients to optimizer parameters."""
        raise NotImplementedError(
            "move_grad_to_optim must be implemented in subclasses."
        )

    def step(
        self,
        step_fn: Callable[..., None],
        is_async: bool = True,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Run an optimizer step using the provided step function.
        The non-async version ensures optimizer updates are complete before returning.
        This ensures every training iteration uses the latest parameters.
        But it will greatly reduce performance, usually not recommended.
        The async version returns immediately after scheduling the step function.
        The training iteration will use 1-step-old parameters, which usually works fine in practice.

        !!! warning
            Data access in the step function should be limited to optimizer parameters only.
            Otherwise, you should be aware of potential data races.

        Args:
            step_fn: Callable that performs an optimization step.
            is_async: Whether to run the step asynchronously.
            *args: Positional arguments forwarded to ``step_fn``.
            **kwargs: Keyword arguments forwarded to ``step_fn``.
        """
        self.optim_updated.wait()  # ensure previous step is done to avoid data race
        for layer_attr in self.layer_attrs:
            layer_attr.param_copied.wait()
            layer_attr.param_copied.clear()
            layer_attr.grad_copied.wait()
            layer_attr.grad_copied.clear()

        if is_async:
            launch_optim_kernel(self.sync_optim_param)
        launch_optim_kernel(self._move_grad_to_optim)
        self.optim_updated.clear()
        launch_optim_kernel(step_fn, *args, **kwargs)
        launch_optim_kernel(lambda: self.optim_updated.set())
        if not is_async:
            self.sync_optim_param()

    def synchronize(self) -> None:
        """Synchronize optimizer parameters and backward gradients
        back to model parameters.
        """
        self.optim_updated.wait()
        for layer_attr in self.layer_attrs:
            layer_attr.param_copied.wait()
            layer_attr.param_copied.clear()

        self.sync_optim_param()
        for layer_attr in self.layer_attrs:
            layer_attr.forward_fence(False)
            layer_attr.backward_fence(False)
        # Move gradients to parameters .grad
        for param in self.model.parameters():
            param_attr = ParamAttribute.get(param)
            grads: List[torch.Tensor] = []
            if param.grad is not None:
                grads.append(param.grad)
            for key, grad in param_attr.grad_cpu.items():
                if grad is not None:
                    grads.append(grad)
                    param_attr.grad_cpu[key] = None
            if len(grads) > 0:
                param.grad = cast(torch.Tensor, sum(grads))


class RoundPipe(RoundPipeBase):
    """Wraps an ``nn.Module`` with RoundPipe's pipelined execution runtime.

    Attributes:
        model_run_config: Default run configuration used when callers do not
            override parameters per invocation.
        layers: Sequence of model layers.
        num_layers: Total number of layers in the pipeline.
        layer_attrs: List of ``LayerAttribute`` storing per-layer events.
        layer_size: List of layer sizes in GB.
        model_timer: ``ModelTimer`` measuring per-layer latency.
    """

    def __init__(
        self,
        model: nn.Module,
        optim_dtype: Optional[torch.dtype] = None,
        name: Optional[str] = None,
        model_run_config: RoundPipeRunConfig = RoundPipeRunConfig(),
        pin_with_register: bool = False,
    ) -> None:
        """Convert model storage to pinned tensors and determine pipeline cuts.

        A nn.Sequential model is split into layers directly. Arbitrary models
        are wrapped as a single layer.

        Args:
            model: Module to wrap. Can be ``nn.Sequential`` or arbitrary model.
            optim_dtype: Data type for optimizer parameters. Defaults to the same
                as param type.
            name: Optional friendly identifier. Defaults to ``file:line``.
            model_run_config: Baseline configuration inherited by invocations.
            pin_with_register: Use cudaHostRegister to pin memory instead of
                torch's pin_memory. This reduces CPU memory usage on very large
                models, but at the cost of ~10% host-device transfer performance.
                Only available with CUDA.
        """
        super().__init__(model, name, optim_dtype)
        self.model_run_config: RoundPipeRunConfig = copy.deepcopy(model_run_config)
        if isinstance(model, nn.Sequential):
            self.layers: List[nn.Module] = list(model)
        else:
            self.layers = [model]

        self.num_layers: int = len(self.layers)
        self.layer_attrs: List[LayerAttribute] = [
            LayerAttribute(f"{self.name}L{i}") for i in range(self.num_layers)
        ]
        self.layer_size: List[float] = [
            get_model_size(layer) / (10**9) for layer in self.layers
        ]
        self.model_timer: ModelTimer = ModelTimer(self.layers)

        for layer in tqdm.tqdm(
            self.layers,
            desc=f"Roundpipe: Process params in {self.name}",
            leave=False,
        ):
            for param in layer.parameters():
                if pin_with_register:
                    pin_tensor(param)
                elif not param.is_pinned():
                    pinned_tensor = torch.empty_like(param.data, pin_memory=True)
                    pinned_tensor.copy_(param.data)
                    param.data = pinned_tensor
                ParamAttribute.set(param, id(layer))
        for buffer in tqdm.tqdm(
            self.model.buffers(),
            total=sum(1 for _ in self.model.buffers()),
            desc=f"Roundpipe: Process buffers in {self.name}",
            leave=False,
        ):
            assert not buffer.requires_grad, "Buffers should not require grad."
            if pin_with_register:
                pin_tensor(buffer)
            elif not buffer.is_pinned():
                pinned_tensor = torch.empty_like(buffer.data, pin_memory=True)
                pinned_tensor.copy_(buffer.data)
                buffer.data = pinned_tensor

        self.RoundPipe_initialized = True

    @override
    def sync_optim_param(self) -> None:
        """Ensure optimizer updated results are copied back to parameters.
        This fuction can run in either the main thread or optimizer thread.
        """
        if not on_optim_stream():
            self.optim_updated.wait()
        # In case of shared params, wait all uploads first
        for layer_attr in self.layer_attrs:
            layer_attr.param_upload_started.wait()
            layer_attr.param_uploaded.synchronize()

        visited_params: Set[int] = set()
        for layer, layer_attr in zip(self.layers, self.layer_attrs):
            for param in layer.parameters():
                param_attr = ParamAttribute.get(param)
                if param_attr.optim is not None and id(param) not in visited_params:
                    param.data.copy_(param_attr.optim)
                    visited_params.add(id(param))

            if on_optim_stream() and layer_attr.param_copied.is_set():
                raise RuntimeError(
                    "param_copied is not cleared as expected, data consistency issue may happen."
                )
            layer_attr.param_copied.set()

    @override
    def _move_grad_to_optim(self) -> None:
        """Move parameter gradients to optimizer parameters.
        This function is designed to run in the optimizer thread only.
        """
        for layer, layer_attr in zip(
            reversed(self.layers),
            reversed(self.layer_attrs),
        ):
            layer_attr.grad_download_started.wait()
            layer_attr.grad_downloaded.synchronize()
            for name, param in layer.named_parameters():
                param_attr = ParamAttribute.get(param)
                grad = param_attr.grad_cpu[id(layer)]
                # Grad buffer keep reference for 1 iteration only
                param_attr.grad_buffer[id(layer)] = grad
                param_attr.grad_cpu[id(layer)] = None
                if grad is None:
                    param_attr.optim_grad_buffer = None
                    continue
                if param_attr.optim is None:
                    raise RuntimeError(
                        f"Parameter {name} has gradient but optimizer data is not "
                        "initialized. This is likely because you did not use optim_parameters() to "
                        "create your optimizer, or you changed parameter requires_grad after optimizer "
                        "creation. Please make sure to create optimizer with optim_parameters()."
                    )

                if param_attr.optim.grad is None:
                    param_attr.optim.grad = param_attr.optim_grad_buffer
                    if param_attr.optim.grad is None:
                        param_attr.optim.grad = torch.empty_like(param_attr.optim)
                    param_attr.optim.grad.copy_(grad)
                else:
                    param_attr.optim.grad.add_(grad.to(dtype=param_attr.optim.dtype))
                param_attr.optim_grad_buffer = param_attr.optim.grad

            if layer_attr.grad_copied.is_set():
                raise RuntimeError(
                    "grad_copied is not cleared as expected, data consistency issue may happen."
                )
            layer_attr.grad_copied.set()

    def forward(
        self,
        *args: Any,
        roundpipe_run_config: RoundPipeRunConfig = RoundPipeRunConfig(),
        **kwargs: Any,
    ) -> Any:
        """Execute a forward pass, optionally enabling gradients per call.

        Args:
            *args: Positional arguments forwarded to the underlying ``model``.
            roundpipe_run_config: Per-call overrides applied on top of the
                model-level run configuration.
            **kwargs: Keyword arguments forwarded to ``model``.

        Returns:
            Output pytree produced by merging or packing all microbatches.

        Raises:
            RuntimeError: If gradients are required but disabled globally.
        """
        full_run_config = FullRoundPipeRunConfig(
            roundpipe_run_config, self.model_run_config
        )
        if full_run_config.requires_grad and not torch.is_grad_enabled():
            raise RuntimeError(
                "RoundPipe model is set to require gradients, but torch gradients are disabled globally."
            )
        batch = Batch(args, kwargs, full_run_config)
        self.model_timer.update_times()
        execute_plan = full_run_config.execute_plan
        if execute_plan is None:
            execute_plan = ModelExecutePlan.auto(
                "train" if full_run_config.requires_grad else "infer", self
            )
        execute_plan.check_valid(
            self.num_layers,
            "train" if full_run_config.requires_grad else "infer",
        )
        timer = IterTimer(self.model_timer)
        tracker = ModelTracker(execute_plan)
        gpu_fwd_layers = [torch.nn.Module() for _ in range(self.num_layers)]
        gpu_bwd_layers = [torch.nn.Module() for _ in range(self.num_layers)]
        run_context = [
            RoundPipeRunContext(
                self,
                gpu_fwd_layers,
                gpu_bwd_layers,
                timer,
                tracker,
                full_run_config.requires_grad,
                i,
                batch.num_microbatch,
                full_run_config.preserve_rng_state,
            )
            for i in range(batch.num_microbatch)
        ]
        for layer_group_id in range(len(tracker.fwd_plan)):
            device = get_next_device()
            device.launch_forward(
                layer_group_id,
                [self.layer_attrs[i] for i in tracker.fwd_plan[layer_group_id]],
                batch,
                run_context,
            )
        tracker.forward_wait_complete(batch.num_microbatch)

        if any(
            isinstance(tensor, torch.Tensor) and tensor.requires_grad
            for batch_output in batch.flatten_states
            for tensor in batch_output
        ):
            if len(tracker.bwd_plan) == 1:
                tag = backward_schedule_simulator.get_next_tag()
                for context in reversed(run_context):
                    tag, output_require_grad_idx, *output_require_grad = cast(
                        Tuple[
                            torch.Tensor, List[int], Unpack[Tuple[torch.Tensor, ...]]
                        ],
                        RoundPipeMicrobatchBackward.apply(
                            context, batch, tag, *context.flatten_inputs[0]
                        ),
                    )
                    for idx, item in zip(output_require_grad_idx, output_require_grad):
                        batch.flatten_states[context.microbatch_id][idx] = item
                backward_schedule_simulator.update_current_tag(tag)
            else:
                gradient_anchor = torch.tensor(
                    0.0, dtype=torch.float32, requires_grad=True
                )
                # ensuring gradients to be calculated even if inputs do not require grad.
                all_inputs = [
                    item
                    for batch_context in run_context
                    for item in batch_context.flatten_inputs[0]
                ]
                output_require_grad_idx, *output_require_grad = cast(
                    Tuple[List[Tuple[int, int]], Unpack[Tuple[torch.Tensor, ...]]],
                    RoundPipeBatchedBackward.apply(
                        run_context, batch, gradient_anchor, *all_inputs
                    ),
                )
                for (batch_idx, idx), item in zip(
                    output_require_grad_idx, output_require_grad
                ):
                    batch.flatten_states[batch_idx][idx] = item

        return batch.dump(full_run_config)

    def forward_backward(
        self,
        input_args: Tuple[Any, ...] = (),
        input_kwargs: Dict[str, Any] = {},
        label: Any = None,
        loss_fn: Callable[
            [Any, Any], Union[Sequence[torch.Tensor], torch.Tensor]
        ] = lambda outputs, labels: outputs,
        return_outputs: bool = False,
        run_config: RoundPipeRunConfig = RoundPipeRunConfig(),
    ) -> Union[
        Tuple[Union[List[torch.Tensor], torch.Tensor], Any],
        List[torch.Tensor],
        torch.Tensor,
    ]:
        """Run a fused forward and backward pass over all microbatches.

        Args:
            input_args: Positional forward arguments.
            input_kwargs: Keyword forward arguments.
            label: Label payload aligned with ``loss_fn`` expectations.
            loss_fn: Callable that consumes ``(outputs, labels)`` and produces
                a loss tensor or sequence of loss tensors.
            return_outputs: Whether to return the model outputs along with loss.
            run_config: Optional per-call overrides for runtime behavior.

        Returns:
            If ``return_outputs`` is ``False``, returns the sum of loss
                tensor(s) across all microbatches.

                If ``return_outputs`` is ``True``, returns a tuple of
                ``(loss_sum, merged_outputs)`` where ``merged_outputs``
                is the output pytree produced by merging or packing all microbatches.

        Raises:
            AssertionError: If gradients are not enabled.
        """
        full_run_config = FullRoundPipeRunConfig(run_config, self.model_run_config)
        assert (
            full_run_config.requires_grad and torch.is_grad_enabled()
        ), "train_iter requires gradients to be enabled."
        batch = Batch(input_args, input_kwargs, full_run_config, label)
        self.model_timer.update_times()
        execute_plan = full_run_config.execute_plan
        if execute_plan is None:
            execute_plan = ModelExecutePlan.auto("fused", self)
        execute_plan.check_valid(self.num_layers, "fused")
        timer = IterTimer(self.model_timer)
        tracker = ModelTracker(execute_plan)
        gpu_fwd_layers = [torch.nn.Module() for _ in range(self.num_layers)]
        gpu_bwd_layers = [torch.nn.Module() for _ in range(self.num_layers)]
        run_context = [
            RoundPipeRunContext(
                self,
                gpu_fwd_layers,
                gpu_bwd_layers,
                timer,
                tracker,
                full_run_config.requires_grad,
                i,
                batch.num_microbatch,
                full_run_config.preserve_rng_state,
            )
            for i in range(batch.num_microbatch)
        ]
        for batch_idx, context in enumerate(run_context):
            context.input_backward_events = batch.backward_events[batch_idx]

        all_inputs = [
            item for batch_input in batch.flatten_states for item in batch_input
        ]
        input_backward_handle = cast(
            torch.Tensor, RoundPipeInputBackward.apply(run_context, *all_inputs)
        )

        for layer_group_id in range(len(tracker.fwd_plan)):
            device = get_next_device()
            device.launch_forward(
                layer_group_id,
                [self.layer_attrs[i] for i in tracker.fwd_plan[layer_group_id]],
                batch,
                run_context,
            )
        device = get_next_device()
        device.launch_forward_backward(
            [self.layer_attrs[i] for i in tracker.bwd_plan[0]],
            batch,
            run_context,
            loss_fn,
            return_outputs,
        )
        for layer_group_id in range(1, len(tracker.bwd_plan)):
            device = get_next_device()
            device.launch_backward(
                layer_group_id,
                [self.layer_attrs[i] for i in tracker.bwd_plan[layer_group_id]],
                run_context,
            )

        tracker.fused_forward_wait_complete(batch.num_microbatch)
        if input_backward_handle.requires_grad:
            tracker.backward_wait_complete(batch.num_microbatch)
            input_backward_handle.backward()

        batch.loss_ready.synchronize()
        if isinstance(batch.loss_list[0], torch.Tensor):
            loss = torch.zeros_like(batch.loss_list[0], device=torch.device("cpu"))
            for batch_loss in batch.loss_list:
                assert isinstance(
                    batch_loss, torch.Tensor
                ), "Inconsistent loss types across microbatches."
                loss = loss + batch_loss
        else:
            loss = [
                torch.zeros_like(t, device=torch.device("cpu"))
                for t in batch.loss_list[0]
            ]
            for batch_loss in batch.loss_list:
                for idx, t in enumerate(batch_loss):
                    loss[idx] = loss[idx] + t

        if return_outputs:
            return loss, batch.dump(full_run_config)
        else:
            return loss


class AutoRoundPipe(RoundPipeBase):
    """Provides partial RoundPipe's features over an arbitrary model.
    This includes optimizer parameter management and async step execution.

    Attributes:
        module_param_uploaded_events: Events signaling params copied to gpu.
            This collects all RoundPipe submodules' event lists.
        module_gradient_ready_events: Events signaling gradients copied to cpu.
            This collects all RoundPipe submodules' event lists.
    """

    def __init__(
        self,
        model: nn.Module,
        name: Optional[str] = None,
        optim_dtype: Optional[torch.dtype] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize AutoRoundPipe over an arbitrary model.

        Args:
            model: Module to wrap.
            name: Optional friendly identifier. Defaults to ``file:line``.
            optim_dtype: Data type for optimizer parameters. Defaults to the same
                as the parameter data type.
            **kwargs: Placeholder for unused keyword arguments.
        """
        super().__init__(model, name, optim_dtype)

        self.virtual_layer_attr = LayerAttribute(f"{self.name}Gbl")
        self.layer_attrs.append(self.virtual_layer_attr)
        for module in self.model.modules():
            if isinstance(module, RoundPipe):
                self.layer_attrs.extend(module.layer_attrs)

        for param in model.parameters():
            if not ParamAttribute.has(param):
                ParamAttribute.set(param, None)

        self.RoundPipe_initialized = True

    @override
    def sync_optim_param(self) -> None:
        """Ensure optimizer updated results are copied back to parameters.
        This fuction can run in either the main thread or optimizer thread.
        """
        if not on_optim_stream():
            self.optim_updated.wait()
        for layer_attr in self.layer_attrs:
            layer_attr.param_upload_started.wait()
            layer_attr.param_uploaded.synchronize()

        for param in self.model.parameters():
            param_attr = ParamAttribute.get(param)
            if param_attr.optim is not None:
                param.data.copy_(param_attr.optim)

        for layer_attr in self.layer_attrs:
            if on_optim_stream() and layer_attr.param_copied.is_set():
                raise RuntimeError(
                    "param_copied is not cleared as expected, data consistency issue may happen."
                )
            layer_attr.param_copied.set()

    @override
    def _move_grad_to_optim(self) -> None:
        """Move parameter gradients to optimizer parameters.
        This function is designed to run in the optimizer thread only.
        """
        for layer_attr in self.layer_attrs:
            layer_attr.grad_download_started.wait()
            layer_attr.grad_downloaded.synchronize()

        for name, param in self.model.named_parameters():
            param_attr = ParamAttribute.get(param)
            grads: List[torch.Tensor] = []
            if param.grad is not None:
                grads.append(param.grad)
                param.grad = None
            for key, grad in param_attr.grad_cpu.items():
                if grad is not None:
                    grads.append(grad)
                # Grad buffer keep reference for 1 iteration only
                param_attr.grad_buffer[key] = grad
                param_attr.grad_cpu[key] = None
            if len(grads) == 0:
                param_attr.optim_grad_buffer = None
                continue
            if param_attr.optim is None:
                raise RuntimeError(
                    f"Parameter {name} has gradient but optimizer data is not "
                    "initialized. This is likely because you did not use optim_parameters() to "
                    "create your optimizer, or you changed parameter requires_grad after optimizer "
                    "creation. Please make sure to create optimizer with optim_parameters()."
                )

            if param_attr.optim.grad is None:
                param_attr.optim.grad = param_attr.optim_grad_buffer
                if param_attr.optim.grad is None:
                    param_attr.optim.grad = torch.empty_like(param_attr.optim)
                param_attr.optim.grad.copy_(grads[0])
            else:
                param_attr.optim.grad.add_(grads[0].to(dtype=param_attr.optim.dtype))
            for grad in grads[1:]:
                param_attr.optim.grad.add_(grad.to(dtype=param_attr.optim.dtype))
            param_attr.optim_grad_buffer = param_attr.optim.grad

        for layer_attr in self.layer_attrs:
            if layer_attr.grad_copied.is_set():
                raise RuntimeError(
                    "grad_copied is not cleared as expected, data consistency issue may happen."
                )
            layer_attr.grad_copied.set()

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Execute a forward pass.

        Args:
            *args: Positional arguments forwarded to the underlying ``model``.
            **kwargs: Keyword arguments forwarded to ``model``.

        Returns:
            Output produced by the underlying ``model``.
        """
        self.virtual_layer_attr.forward_fence(False)
        ret = self.model(*args, **kwargs)
        self.virtual_layer_attr.backward_fence(False)
        return ret
