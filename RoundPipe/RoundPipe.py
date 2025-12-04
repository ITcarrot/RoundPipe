"""The RoundPipe model wrapper and execution runtime."""

from beartype.typing import * # type: ignore[reportWildcardImportFromLibrary]
from beartype import beartype
import traceback
import copy

import tqdm
import torch
import torch.nn as nn

from .batch import Batch
from .device import get_next_device
from .run import RoundPipeRunContext, RoundPipeBatchedBackward, RoundPipeMicrobatchBackward, RoundPipeInputBackward
from .RunConfig import RoundPipeRunConfig, FullRoundPipeRunConfig
from .scheduler import ModelExecutePlan, backward_schedule_simulator
from .timer import ModelTimer
from .utils import get_model_size

@beartype
class RoundPipe(nn.Module):
    """Wraps an ``nn.Module`` with RoundPipe's pipelined execution runtime.

    Attributes:
        name: Human-readable identifier shown in traces/logs.
        model: The provided module wrapped for RoundPipe execution.
        original_model: Reference to the pre-wrapped module when shimming
            attribute access.
        use_fp16: Flag indicating whether weights/buffers are stored as FP16.
        model_run_config: Default run configuration used when callers do not
            override parameters per invocation.
        layers: Sequence of logical pipeline stages.
        num_layers: Total number of layer groups in the pipeline.
        layer_workload: Estimated byte size per layer, used for scheduling.
        layer_gradient_ready_events: CUDA events used to gate gradient reads.
        model_timer: ``ModelTimer`` measuring per-layer latency.
        RoundPipe_initialized: Guard marker to customize attribute behavior.
    """
    def __init__(self,
                 model: nn.Module,
                 use_fp16: bool = False,
                 name: Optional[str] = None,
                 model_run_config: RoundPipeRunConfig = RoundPipeRunConfig()) -> None:
        """Convert model storage to pinned tensors and determine pipeline cuts.
        
        A nn.Sequential model is split into layers directly. Arbitrary models
        are wrapped as a single layer.

        Args:
            model: Module to wrap. Can be ``nn.Sequential`` or arbitrary model.
            use_fp16: Whether to store floating parameters/buffers in FP16.
            name: Optional friendly identifier. Defaults to ``file:line``.
            model_run_config: Baseline configuration inherited by invocations.
        """
        super().__init__()
        filename, lineno, _, _ = traceback.extract_stack()[-3]
        self.name: str = name if name else f'{filename.split("/")[-1]}:{lineno}'
        self.model: nn.Module = model
        self.original_model: Optional[nn.Module] = None # placeholder for original model if needing its functions
        self.use_fp16: bool = use_fp16
        self.model_run_config: RoundPipeRunConfig = copy.deepcopy(model_run_config)
        if isinstance(model, nn.Sequential):
            self.layers: List[nn.Module] = list(model)
        else:
            self.layers = [model]

        self.num_layers: int = len(self.layers)
        self.layer_workload: List[float] = []
        self.layer_gradient_ready_events: List[torch.cuda.Event] = [torch.cuda.Event() for _ in range(self.num_layers)] # type: ignore[reportAttributeAccessIssue]
        for layer in self.layers:
            self.layer_workload.append(get_model_size(layer))
        self.model_timer: ModelTimer = ModelTimer(self.num_layers)

        for parm in tqdm.tqdm(self.model.parameters(), total=sum(1 for _ in self.model.parameters()),
                              desc=f'Roundpipe: Process params in {self.name}', leave=False):
            pinned_tensor = torch.empty_like(parm.data, dtype=torch.float16 if use_fp16 and parm.is_floating_point() else None, pin_memory=True)
            pinned_tensor.copy_(parm.data)
            parm.data = pinned_tensor
            parm.data_cpu = pinned_tensor # type: ignore[attr-defined]
        for buffer in tqdm.tqdm(self.model.buffers(), total=sum(1 for _ in self.model.buffers()),
                                desc=f'Roundpipe: Process buffers in {self.name}', leave=False):
            pinned_tensor = torch.empty_like(buffer.data, dtype=torch.float16 if use_fp16 and buffer.is_floating_point() else None, pin_memory=True)
            pinned_tensor.copy_(buffer.data)
            buffer.data = pinned_tensor
            buffer.data_cpu = pinned_tensor # type: ignore[attr-defined]

        self.RoundPipe_initialized: bool = True

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
                        = RoundPipeMicrobatchBackward.apply(context, batch, tag, *context.flatten_inputs[0]) # type: ignore
                    for idx, item in zip(output_require_grad_idx, output_require_grad):
                        batch.flatten_states[context.microbatch_id][idx] = item
                backward_schedule_simulator.update_current_tag(tag)
            else:
                gradient_anchor = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
                # ensuring gradients to be calculated even if inputs do not require grad.
                all_inputs = [item for batch_context in run_context
                            for item in batch_context.flatten_inputs[0]]
                output_require_grad_idx, *output_require_grad = RoundPipeBatchedBackward.apply(run_context, batch, gradient_anchor, *all_inputs) # type: ignore
                for (batch_idx, idx), item in zip(output_require_grad_idx, output_require_grad):
                    batch.flatten_states[batch_idx][idx] = item

        return batch.dump(full_run_config)

    def train_iter(self, input_args: Tuple[Any, ...] = (),
                   input_kwargs: Dict[str, Any] = {},
                   label: Any = None,
                   loss_fn: Callable[[Any, Any], Union[Sequence[torch.Tensor], torch.Tensor]] = lambda outputs, labels: outputs,
                   run_config: RoundPipeRunConfig = RoundPipeRunConfig()
                   ) -> Tuple[Union[List[torch.Tensor], torch.Tensor], Any]:
        """Run one training iteration.

        Args:
            input_args: Positional forward arguments.
            input_kwargs: Keyword forward arguments.
            label: Label payload aligned with ``loss_fn`` expectations.
            loss_fn: Callable that consumes ``(outputs, labels)`` and produces
                a loss tensor or sequence of loss tensors.
            run_config: Optional per-call overrides for runtime behavior.

        Returns:
            The sum of all microbatch losses used for backward
            The merged or packed outputs from all microbatches.

        Raises:
            AssertionError: If gradients are not enabled.
        """
        full_run_config = FullRoundPipeRunConfig(run_config, self.model_run_config)
        assert full_run_config.requires_grad and torch.is_grad_enabled(), \
               "train_iter requires gradients to be enabled."
        batch = Batch(input_args, input_kwargs, full_run_config, label)
        execute_plan = ModelExecutePlan(self, True)
        run_context = [RoundPipeRunContext(self, execute_plan, full_run_config.requires_grad,
                                           i, batch.num_microbatch, full_run_config.preserve_rng_state)
                       for i in range(batch.num_microbatch)]
        for batch_idx, context in enumerate(run_context):
            context.input_backward_events = batch.backward_events[batch_idx]

        all_inputs = [item for batch_input in batch.flatten_states for item in batch_input]
        input_backward_handle: torch.Tensor = RoundPipeInputBackward.apply(run_context, *all_inputs) # type: ignore

        for layer_group_id in range(len(execute_plan.fwd_plan)):
            device = get_next_device()
            device.launch_forward(layer_group_id, batch, run_context)
        execute_plan.forward_wait_complete(batch.num_microbatch)
        device = get_next_device()
        device.launch_forward_backward(batch, run_context, loss_fn)
        for layer_group_id in range(1, len(execute_plan.bwd_plan)):
            device = get_next_device()
            device.launch_backward(layer_group_id, run_context)
        execute_plan.backward_wait_complete(batch.num_microbatch)

        if input_backward_handle.requires_grad:
            input_backward_handle.backward()
        if isinstance(batch.loss_list[0], torch.Tensor):
            loss = torch.zeros_like(batch.loss_list[0], device=torch.device('cpu'))
            for batch_loss in batch.loss_list:
                loss = loss + batch_loss.cpu() # type: ignore[reportOperatorIssue]
        else:
            loss = [torch.zeros_like(t, device=torch.device('cpu')) for t in batch.loss_list[0]]
            for batch_loss in batch.loss_list:
                for idx, t in enumerate(batch_loss):
                    loss[idx] = loss[idx] + t.cpu()
        return loss, batch.dump(full_run_config)
