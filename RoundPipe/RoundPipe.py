"""The RoundPipe model wrapper and execution runtime."""

from typing_extensions import *
import traceback
import copy
import warnings
import threading

import tqdm
import torch
import torch.nn as nn

from .batch import Batch
from .context import doing_optimizer
from .device import get_next_device
from .optim_stream import launch_optim_kernel, on_optim_stream
from .param import ParamAttribute
from .run import RoundPipeRunContext, RoundPipeBatchedBackward, RoundPipeMicrobatchBackward, RoundPipeInputBackward
from .RunConfig import RoundPipeRunConfig, FullRoundPipeRunConfig
from .scheduler import ModelExecutePlan, backward_schedule_simulator
from .timer import ModelTimer
from .utils import get_model_size

class RoundPipeBase(nn.Module):
    """Common attributes and methods of RoundPipe and AutoRoundPipe

    Attributes:
        name: Human-readable identifier shown in traces/logs.
        model: The provided module wrapped for RoundPipe execution.
        original_model: Reference to the pre-wrapped module when shimming
            attribute access.
        optim_dtype: Data type for optimizer parameters.
        optim_updated: Event signaling optimizer have updated.
    """
    def __init__(self, model: nn.Module, name: Optional[str] = None,
                 optim_dtype: Optional[torch.dtype] = None) -> None:
        '''Initialize the RoundPipe base wrapper.
        
        Args:
            model: Module to wrap.
            name: Optional friendly identifier. Defaults to ``file:line``.
            optim_dtype: Data type for optimizer parameters. Defaults to the same
                as the parameter data type.
        '''
        super().__init__()
        # call stack: beartype -> (Auto)RoundPipe -> beartype -> RoundPipeBase
        filename, lineno, _, _ = traceback.extract_stack()[-5]
        self.name: str = name if name else f'{filename.split("/")[-1]}:{lineno}'
        self.model: nn.Module = model
        self.original_model: Optional[nn.Module] = None # placeholder for original model if needing its functions

        self.optim_dtype: Optional[torch.dtype] = optim_dtype
        self.optim_updated: threading.Event = threading.Event()
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
        if 'RoundPipe_initialized' in self.__dict__:
            if self.original_model is not None:
                setattr(self.original_model, name, value)
            setattr(self.model, name, value)
        else:
            super().__setattr__(name, value)

    def __delattr__(self, name: str) -> None:
        """Ensure attribute deletions propagate to wrapped/original modules."""
        if 'RoundPipe_initialized' in self.__dict__:
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
        object.__setattr__(self, 'original_model', original_model)

    def named_parameters(self, prefix: str = "", recurse: bool = True,
                         remove_duplicate: bool = True) -> Iterator[tuple[str, nn.Parameter]]:
        """Iterator over named parameters. Overrides to warn against direct use,
            and redirect to optim_named_parameters under optimizer context.
        """
        if doing_optimizer() and recurse:
            return cast(Iterator[tuple[str, nn.Parameter]],
                        self.optim_named_parameters(prefix, remove_duplicate))
        warnings.warn("RoundPipe will manage parameter location and dtype internally. "
                      "\nAccessing parameters() or named_parameters() directly may "
                      "lead to unexpected behavior. \nIf you intend to get parameters "
                      "for optimization, please use optim_parameters() or "
                      "optim_named_parameters() instead.", UserWarning)
        return super().named_parameters(prefix, recurse, remove_duplicate)

    def parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]:
        """Iterator over parameters. Overrides to redirect to optim_parameters
            under optimizer context.
        """
        if doing_optimizer() and recurse:
            return cast(Iterator[nn.Parameter], self.optim_parameters())
        return super().parameters(recurse)

    def optim_named_parameters(self, prefix: str = "", remove_duplicate: bool = True
                               ) -> Iterator[tuple[str, torch.Tensor]]:
        """Iterator over named parameters suitable for optimizer consumption.

        Args:
            prefix: Prefix to prepend to parameter names.
            remove_duplicate: Whether to skip duplicate parameters.

        Yields:
            Tuples of parameter names and their optimizer-ready tensors.
        """
        for name, parm in super().named_parameters(prefix, True, remove_duplicate):
            parm_attr = ParamAttribute.get(parm)
            if not parm_attr.optim_inited() and parm.requires_grad:
                parm_attr.data_optim = parm_attr.data_cpu.to(dtype=self.optim_dtype, copy=True)
            yield name, parm_attr.data_optim
 
    def optim_parameters(self) -> Iterator[torch.Tensor]:
        """Iterator over parameters suitable for optimizer consumption.

        Yields:
            Parameters stored in their optimizer-ready format.
        """
        for _, parm in self.optim_named_parameters():
            yield parm

    def lock_param(self) -> None:
        """Lock parameter data during optimizer -> parameter copy"""
        raise NotImplementedError("lock_param only available in class RoundPipe.")

    def sync_optim_param(self) -> None:
        """Ensure optimizer updated results are copied back to parameters."""
        raise NotImplementedError("sync_optim_param must be implemented in subclasses.")
    
    def move_grad_to_optim(self) -> None:
        """Move parameter gradients to optimizer parameters."""
        raise NotImplementedError("move_grad_to_optim must be implemented in subclasses.")

    def lock_grad(self) -> None:
        """Lock parameter gradients and transfer events to avoid being
        overwritten by next backward.
        """
        raise NotImplementedError("lock_grad must be implemented in subclasses.")

    def step(self, step_fn: Callable[..., None], is_async: bool = True,
             *args: Any, **kwargs: Any) -> None:
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
        self.optim_updated.wait() # ensure previous step is done to avoid data race
        self.lock_grad()
        for name, param in super().named_parameters():
            param_attr = ParamAttribute.get(param)
            if param.grad is not None and not param_attr.optim_inited():
                raise RuntimeError(f"Parameter {name} has gradient but optimizer data is not "
                    "initialized. This is likely because you did not use optim_parameters() to "
                    "create your optimizer, or you changed parameter requires_grad after optimizer "
                    "creation. Please make sure to create optimizer with optim_parameters().")
            # Move grad reference away from param to avoid accidental modification
            param_attr.data_grad = param.grad
            param.grad = None

        if is_async:
            if isinstance(self, RoundPipe):
                self.lock_param()
                launch_optim_kernel(self.sync_optim_param)
            else:
                self.sync_optim_param()
        launch_optim_kernel(self.move_grad_to_optim)
        self.optim_updated.clear()
        launch_optim_kernel(step_fn, *args, **kwargs)
        launch_optim_kernel(lambda: self.optim_updated.set())
        if not is_async:
            self.sync_optim_param()

class RoundPipe(RoundPipeBase):
    """Wraps an ``nn.Module`` with RoundPipe's pipelined execution runtime.

    Attributes:
        model_run_config: Default run configuration used when callers do not
            override parameters per invocation.
        layers: Sequence of logical pipeline stages.
        num_layers: Total number of layer groups in the pipeline.
        layer_workload: Estimated byte size per layer, used for scheduling.
        layer_param_copied: Event signaling a new version of parameters
            has been copied from optimizer.
        layer_param_uploaded_events: Event signaling params is copied to gpu.
        layer_gradient_ready_events: Event signaling gradient is copied to cpu.
        layer_gradient_copied: Event signaling gradient has been moved to optimizer.
            This can avoid allocating doubled gradient memory on cpu. When this is
            set, it implies the ParamAttribute.data_grad can be reused for next backward.
        model_timer: ``ModelTimer`` measuring per-layer latency.
    """
    def __init__(self,
                 model: nn.Module,
                 optim_dtype: Optional[torch.dtype] = None,
                 name: Optional[str] = None,
                 model_run_config: RoundPipeRunConfig = RoundPipeRunConfig()) -> None:
        """Convert model storage to pinned tensors and determine pipeline cuts.
        
        A nn.Sequential model is split into layers directly. Arbitrary models
        are wrapped as a single layer.

        Args:
            model: Module to wrap. Can be ``nn.Sequential`` or arbitrary model.
            optim_dtype: Data type for optimizer parameters. Defaults to the same
                as param type.
            name: Optional friendly identifier. Defaults to ``file:line``.
            model_run_config: Baseline configuration inherited by invocations.
        """
        super().__init__(model, name, optim_dtype)
        self.model_run_config: RoundPipeRunConfig = copy.deepcopy(model_run_config)
        if isinstance(model, nn.Sequential):
            self.layers: List[nn.Module] = list(model)
        else:
            self.layers = [model]

        self.num_layers: int = len(self.layers)
        self.layer_workload: List[float] = []
        for layer in self.layers:
            self.layer_workload.append(get_model_size(layer))
        self.model_timer: ModelTimer = ModelTimer(self.layers)

        self.layer_param_copied: List[threading.Event] = [threading.Event() for _ in range(self.num_layers)]
        for e in self.layer_param_copied:
            e.set()
        self.layer_param_uploaded_events: List[torch.cuda.Event] \
            = [cast(torch.cuda.Event, torch.cuda.Event()) for _ in range(self.num_layers)]
        self.layer_gradient_ready_events: List[torch.cuda.Event] \
            = [cast(torch.cuda.Event, torch.cuda.Event()) for _ in range(self.num_layers)]
        self.layer_gradient_copied: List[threading.Event] = [threading.Event() for _ in range(self.num_layers)]
        for e in self.layer_gradient_copied:
            e.set()

        for parm in tqdm.tqdm(self.model.parameters(), total=sum(1 for _ in self.model.parameters()),
                              desc=f'Roundpipe: Process params in {self.name}', leave=False):
            pinned_tensor = torch.empty_like(parm.data, pin_memory=True)
            pinned_tensor.copy_(parm.data)
            parm.data = pinned_tensor
            ParamAttribute.set(parm)
        for buffer in tqdm.tqdm(self.model.buffers(), total=sum(1 for _ in self.model.buffers()),
                                desc=f'Roundpipe: Process buffers in {self.name}', leave=False):
            pinned_tensor = torch.empty_like(buffer.data, pin_memory=True)
            pinned_tensor.copy_(buffer.data)
            buffer.data = pinned_tensor
            ParamAttribute.set(buffer)

        self.RoundPipe_initialized = True

    @override
    def lock_param(self) -> None:
        """Lock parameter data during optimizer -> parameter copy"""
        for event in self.layer_param_copied:
            event.clear()

    @override
    def sync_optim_param(self) -> None:
        """Ensure optimizer updated results are copied back to parameters.
        This fuction can run in either the main thread or optimizer thread.
        """
        if not on_optim_stream():
            self.optim_updated.wait()
        for layer, event, copied_event in zip(self.layers, self.layer_param_uploaded_events, self.layer_param_copied):
            event.synchronize()
            if on_optim_stream():
                assert not copied_event.is_set(), "lock_param must be called before syncing optim param."
            for param in layer.parameters():
                param_attr = ParamAttribute.get(param)
                if param_attr.optim_inited():
                    param_attr.data_cpu.copy_(param_attr.data_optim)
            if on_optim_stream():
                copied_event.set()

    @override
    def move_grad_to_optim(self) -> None:
        """Move parameter gradients to optimizer parameters.
        This function is designed to run in the optimizer thread only.
        """
        for layer, ready_event, copied_event in zip(
            reversed(self.layers), reversed(self.layer_gradient_ready_events), reversed(self.layer_gradient_copied)):
            assert not copied_event.is_set(), "lock_grad must be called before moving grad to optim."
            ready_event.synchronize()
            for param in layer.parameters():
                param_attr = ParamAttribute.get(param)
                if param_attr.data_grad is not None:
                    if param_attr.data_optim.grad is None:
                        if param_attr.optim_grad is not None:
                            param_attr.data_optim.grad = param_attr.optim_grad
                        else:
                            param_attr.data_optim.grad = torch.empty_like(param_attr.data_optim)
                        param_attr.data_optim.grad.copy_(param_attr.data_grad)
                    else:
                        param_attr.data_optim.grad.add_(param_attr.data_grad.to(dtype=param_attr.data_optim.dtype))
                    param_attr.optim_grad = param_attr.data_optim.grad
                else:
                    param_attr.optim_grad = None # clear optim_grad reference if no grad
            copied_event.set()

    @override
    def lock_grad(self) -> None:
        """Lock parameter gradients and transfer events to avoid being
        overwritten by next backward.
        """
        for copied_event in self.layer_gradient_copied:
            copied_event.clear()

    def forward(self, *args: Any,
                roundpipe_run_config: RoundPipeRunConfig = RoundPipeRunConfig(), **kwargs: Any) -> Any:
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
        full_run_config = FullRoundPipeRunConfig(roundpipe_run_config, self.model_run_config)
        if full_run_config.requires_grad and not torch.is_grad_enabled():
            raise RuntimeError("RoundPipe model is set to require gradients, but torch gradients are disabled globally.")
        batch = Batch(args, kwargs, full_run_config)
        self.model_timer.update_times()
        execute_plan = ModelExecutePlan(self, False)
        run_context = [RoundPipeRunContext(self, execute_plan, full_run_config.requires_grad,
                                           i, batch.num_microbatch, full_run_config.preserve_rng_state)
                       for i in range(batch.num_microbatch)]
        for layer_group_id in range(len(execute_plan.fwd_plan)):
            device = get_next_device()
            device.launch_forward(layer_group_id, batch, run_context)
        execute_plan.forward_wait_complete(batch.num_microbatch)
        
        if any(isinstance(tensor, torch.Tensor) and tensor.requires_grad
               for batch_output in batch.flatten_states
               for tensor in batch_output):
            if len(execute_plan.bwd_plan) == 1:
                tag = backward_schedule_simulator.get_next_tag()
                for context in reversed(run_context):
                    tag, output_require_grad_idx, *output_require_grad \
                        = cast(Tuple[torch.Tensor, List[int], Unpack[Tuple[torch.Tensor, ...]]],
                          RoundPipeMicrobatchBackward.apply(context, batch, tag, *context.flatten_inputs[0]))
                    for idx, item in zip(output_require_grad_idx, output_require_grad):
                        batch.flatten_states[context.microbatch_id][idx] = item
                backward_schedule_simulator.update_current_tag(tag)
            else:
                gradient_anchor = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
                # ensuring gradients to be calculated even if inputs do not require grad.
                all_inputs = [item for batch_context in run_context
                            for item in batch_context.flatten_inputs[0]]
                output_require_grad_idx, *output_require_grad \
                    = cast(Tuple[List[Tuple[int, int]], Unpack[Tuple[torch.Tensor, ...]]],
                      RoundPipeBatchedBackward.apply(run_context, batch, gradient_anchor, *all_inputs))
                for (batch_idx, idx), item in zip(output_require_grad_idx, output_require_grad):
                    batch.flatten_states[batch_idx][idx] = item

        return batch.dump(full_run_config)

    def forward_backward(self, input_args: Tuple[Any, ...] = (),
                         input_kwargs: Dict[str, Any] = {},
                         label: Any = None,
                         loss_fn: Callable[[Any, Any], Union[Sequence[torch.Tensor], torch.Tensor]] = lambda outputs, labels: outputs,
                         return_outputs: bool = False,
                         run_config: RoundPipeRunConfig = RoundPipeRunConfig()
                         ) -> Union[Tuple[Union[List[torch.Tensor], torch.Tensor], Any],
                                    List[torch.Tensor], torch.Tensor]:
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
        assert full_run_config.requires_grad and torch.is_grad_enabled(), \
               "train_iter requires gradients to be enabled."
        batch = Batch(input_args, input_kwargs, full_run_config, label)
        self.model_timer.update_times()
        execute_plan = ModelExecutePlan(self, True)
        run_context = [RoundPipeRunContext(self, execute_plan, full_run_config.requires_grad,
                                           i, batch.num_microbatch, full_run_config.preserve_rng_state)
                       for i in range(batch.num_microbatch)]
        for batch_idx, context in enumerate(run_context):
            context.input_backward_events = batch.backward_events[batch_idx]

        all_inputs = [item for batch_input in batch.flatten_states for item in batch_input]
        input_backward_handle = cast(torch.Tensor, RoundPipeInputBackward.apply(run_context, *all_inputs))

        for layer_group_id in range(len(execute_plan.fwd_plan)):
            device = get_next_device()
            device.launch_forward(layer_group_id, batch, run_context)
        execute_plan.forward_wait_complete(batch.num_microbatch)
        device = get_next_device()
        device.launch_forward_backward(batch, run_context, loss_fn, return_outputs)
        for layer_group_id in range(1, len(execute_plan.bwd_plan)):
            device = get_next_device()
            device.launch_backward(layer_group_id, run_context)
        execute_plan.backward_wait_complete(batch.num_microbatch)

        if input_backward_handle.requires_grad:
            input_backward_handle.backward()

        batch.loss_ready.synchronize()
        if isinstance(batch.loss_list[0], torch.Tensor):
            loss = torch.zeros_like(batch.loss_list[0], device=torch.device('cpu'))
            for batch_loss in batch.loss_list:
                assert isinstance(batch_loss, torch.Tensor), \
                    "Inconsistent loss types across microbatches."
                loss = loss + batch_loss
        else:
            loss = [torch.zeros_like(t, device=torch.device('cpu')) for t in batch.loss_list[0]]
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
    def __init__(self,
                 model: nn.Module,
                 name: Optional[str] = None,
                 optim_dtype: Optional[torch.dtype] = None,
                 **kwargs: Any) -> None:
        """Initialize AutoRoundPipe over an arbitrary model.
        
        Args:
            model: Module to wrap.
            name: Optional friendly identifier. Defaults to ``file:line``.
            optim_dtype: Data type for optimizer parameters. Defaults to the same
                as the parameter data type.
            **kwargs: Placeholder for unused keyword arguments.
        """
        super().__init__(model, name, optim_dtype)
        self.module_param_uploaded_events: List[List[torch.cuda.Event]] = []
        self.module_gradient_ready_events: List[List[torch.cuda.Event]] = []
        self.module_gradient_copied: List[List[threading.Event]] = []

        for module in self.model.modules():
            if isinstance(module, RoundPipe):
                self.module_param_uploaded_events.append(module.layer_param_uploaded_events)
                self.module_gradient_ready_events.append(module.layer_gradient_ready_events)
                self.module_gradient_copied.append(module.layer_gradient_copied)

        for param in model.parameters():
            if not ParamAttribute.has(param):
                ParamAttribute.set(param)

        self.RoundPipe_initialized = True

    @override
    def sync_optim_param(self) -> None:
        """Ensure optimizer updated results are copied back to parameters."""
        self.optim_updated.wait()
        for layer_events in self.module_param_uploaded_events:
            for event in layer_events:
                event.synchronize()
        for param in self.model.parameters():
            parm_attr = ParamAttribute.get(param)
            if parm_attr.optim_inited():
                parm_attr.data_cpu.copy_(parm_attr.data_optim)

    @override
    def move_grad_to_optim(self) -> None:
        """Move parameter gradients to optimizer parameters.
        This function is designed to run in the optimizer thread only.
        """
        for layer_events in self.module_gradient_ready_events:
            for event in layer_events:
                event.synchronize()
        for layer_copied_events in self.module_gradient_copied:
            for copied_event in layer_copied_events:
                assert not copied_event.is_set(), "lock_grad must be called before moving grad to optim."

        for param in self.model.parameters():
            param_attr = ParamAttribute.get(param)
            if param_attr.data_grad is not None:
                if param_attr.data_optim.grad is None:
                    if param_attr.optim_grad is not None:
                        param_attr.data_optim.grad = param_attr.optim_grad
                    else:
                        param_attr.data_optim.grad = torch.empty_like(param_attr.data_optim)
                    param_attr.data_optim.grad.copy_(param_attr.data_grad)
                else:
                    param_attr.data_optim.grad.add_(param_attr.data_grad.to(dtype=param_attr.data_optim.dtype))
                param_attr.optim_grad = param_attr.data_optim.grad
            else:
                param_attr.optim_grad = None # clear optim_grad reference if no grad

        for layer_copied_events in self.module_gradient_copied:
            for copied_event in layer_copied_events:
                copied_event.set()

    @override
    def lock_grad(self) -> None:
        """Lock parameter gradients and transfer events to avoid being
        overwritten by next backward.
        """
        for layer_copied_events in self.module_gradient_copied:
            for copied_event in layer_copied_events:
                copied_event.clear()

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Execute a forward pass.

        Args:
            *args: Positional arguments forwarded to the underlying ``model``.
            **kwargs: Keyword arguments forwarded to ``model``.

        Returns:
            Output produced by the underlying ``model``.
        """
        return self.model(*args, **kwargs)
