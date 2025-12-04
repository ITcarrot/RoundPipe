"""Utilities for handling microbatches and host/device synchronization.

The helpers in this module are consumed by ``RoundPipe`` to convert arbitrary
argument trees into iterable structures, split them into microbatches, keep
track of transfer events, and merge results back on the host once transfers
finish.

Attributes:
    avg_reducer: Predefined reducer that averages scalar losses across microbatches.
"""

from beartype.typing import * # type: ignore[reportWildcardImportFromLibrary]
import warnings

import torch
from torch.distributed.pipelining.microbatch import TensorChunkSpec, _Replicate, split_args_kwargs_into_chunks, merge_chunks, _CustomReducer
from torch.utils._pytree import tree_flatten, tree_unflatten, TreeSpec

from .RunConfig import FullRoundPipeRunConfig
from .transfer import PinnedUpload, RegisterBackwardEvent

class RoundPipePackedData(list):
    """Container that couples host tensors with CUDA transfer markers.

    Attributes:
        transfer_event: List of ``(forward_event, backward_event)`` tuples
            per microbatch that signals when the data and, optionally,
            gradients are ready on the host.
    """

    def __init__(self, data: List[Any],
                 transfer_event: List[Tuple[torch.cuda.Event, torch.cuda.Event]]) -> None:
        """Build the packed container.

        Args:
            data: Microbatch outputs stored on the host.
            transfer_event: CUDA events marking forward/ backward readiness for
                each microbatch result.
        """
        super().__init__(data)
        self.transfer_event: List[Tuple[torch.cuda.Event, torch.cuda.Event]] = transfer_event

    def synchronize(self) -> None:
        """Block until all hosted tensors are fully transferred.

        Returns:
            The call completes only after forward events finish.
        """
        for forward_event, _ in self.transfer_event:
            forward_event.synchronize()

def guess_split_spec(data: Any, expected_batchsize: Optional[int] = None) -> Tuple[Any, Optional[int]]:
    """Infer how ``torch.distributed`` should chunk a nested argument tree.

    Args:
        data: Arbitrary pytree holding tensors and Python objects.
        expected_batchsize: Optional batch-size hint used to identify tensors
            that should be chunked along dimension 0.

    Returns:
        A tuple ``(spec, inferred_batch_size)`` where ``spec`` mirrors
            ``data`` and stores ``TensorChunkSpec`` or ``_Replicate`` entries,
            and ``inferred_batch_size`` is the common chunkable size if one could
            be identified, otherwise ``None``.
    """
    flatten, flatten_spec = tree_flatten(data)
    guessed_spec = []
    maybe_batchsize = []
    for item in flatten:
        if isinstance(item, torch.Tensor) and item.ndim > 0 and (expected_batchsize is None or item.size(0) == expected_batchsize):
            guessed_spec.append(TensorChunkSpec(0))
            maybe_batchsize.append(item.size(0))
        else:
            guessed_spec.append(_Replicate)
            if isinstance(item, RoundPipePackedData) \
                and all(isinstance(batch_item, torch.Tensor) for batch_item in item) \
                and all(batch_item.ndim > 0 for batch_item in item):
                maybe_batchsize.append(sum(batch_item.size(0) for batch_item in item))
    if len(maybe_batchsize) == 0 or any(bs != maybe_batchsize[0] for bs in maybe_batchsize):
        guessed_batchsize = None
    else:
        guessed_batchsize = maybe_batchsize[0]
    return tree_unflatten(guessed_spec, flatten_spec), guessed_batchsize

def get_avg_reducer_args() -> Tuple[torch.Tensor, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]:
    """Build a reducer that averages scalar losses across microbatches.

    Returns:
        the initial accumulator tensor
        a reducer callable that updates the running average.
    """
    init_val = torch.tensor(0)
    init_val.roundpipe_avg_reducer_sum = torch.tensor(0) # type: ignore[reportAttributeAccessIssue]
    init_val.roundpipe_avg_reducer_count = 0 # type: ignore[reportAttributeAccessIssue]
    def reduce(reduced_val: torch.Tensor, new_val: torch.Tensor) -> torch.Tensor:
        val_sum = reduced_val.roundpipe_avg_reducer_sum + new_val # type: ignore[reportAttributeAccessIssue]
        val_count = reduced_val.roundpipe_avg_reducer_count + 1 # type: ignore[reportAttributeAccessIssue]
        new_reduced = val_sum / val_count
        new_reduced.roundpipe_avg_reducer_sum = val_sum # type: ignore[reportAttributeAccessIssue]
        new_reduced.roundpipe_avg_reducer_count = val_count # type: ignore[reportAttributeAccessIssue]
        return new_reduced
    return init_val, reduce
avg_reducer: _CustomReducer = _CustomReducer(*get_avg_reducer_args())

class Batch:
    """Holds flattened microbatch state, labels, and CUDA events.

    Attributes:
        flatten_states: Flattened argument tensors per microbatch.
        flatten_specs: ``TreeSpec`` objects per microbatch for reconstruction.
        forward_events: CUDA events signaling when forward transfers complete.
        backward_events: CUDA events signaling when gradients arrive on host.
        num_microbatch: Actual number of microbatches generated.
        label_list: Labels aligned with each microbatch.
        loss_list: Loss tensors accumulated per microbatch.
    """

    def __init__(self, args: Tuple, kwargs: Dict[str, Any],
                 run_config: FullRoundPipeRunConfig,
                 label: Any = None) -> None:
        """Split inputs and reconcile chained ``RoundPipePackedData`` sources.

        Args:
            args: Positional arguments provided to the wrapped model.
            kwargs: Keyword arguments provided to the wrapped model.
            run_config: Effective runtime configuration.
            label: Optional label payload for training flows.

        Raises:
            AssertionError: If custom splitters return malformed structures.
        """
        if run_config.num_microbatch == 1:
            args_list, kwargs_list = [args], [kwargs]
        elif callable(run_config.split_input):
            args_list, kwargs_list = run_config.split_input(args, kwargs, run_config.num_microbatch)
            assert isinstance(args_list, list) and isinstance(kwargs_list, list) \
                   and len(args_list) == len(kwargs_list), \
                   "split_input function must return two lists of equal length."
        else:
            args_spec, kwargs_spec = run_config.split_input
            guessed_batchsize = None
            if args_spec is None:
                args_spec, guessed_batchsize = guess_split_spec(args)
            if kwargs_spec is None:
                kwargs_spec, guessed_batchsize = guess_split_spec(kwargs, guessed_batchsize)
            args_list, kwargs_list = split_args_kwargs_into_chunks(
                args, kwargs, run_config.num_microbatch, args_spec, kwargs_spec)

        self.flatten_states: List[List[Any]] = []
        self.flatten_specs: List[TreeSpec] = []
        self.forward_events: List[Sequence[torch.cuda.Event]] = []
        self.backward_events: List[Sequence[torch.cuda.Event]] = []
        for batch_idx, args_kwargs in enumerate(zip(args_list, kwargs_list)):
            forward_event: Set[torch.cuda.Event] = set()
            backward_event: Set[torch.cuda.Event] = set()
            cpu_tensor_backward_event: torch.cuda.Event = torch.cuda.Event() # type: ignore[reportAssignmentType]
            flatten_input, flatten_spec = tree_flatten(args_kwargs)
            for idx, item in enumerate(flatten_input):
                if isinstance(item, torch.Tensor):
                    assert item.is_cpu, 'All inputs to RoundPipe must be on CPU.'
                    # If the input tensor requires gradients, register a autograd
                    # graph node to ensure the the gradient is transfered back to
                    # CPU before using it in the subsequent computation.
                    if item.requires_grad:
                        flatten_input[idx] = RegisterBackwardEvent.apply(item, cpu_tensor_backward_event)
                        backward_event.add(cpu_tensor_backward_event)
            try:
                for idx, item in enumerate(flatten_input):
                    if isinstance(item, RoundPipePackedData):
                        # Each packed tensor carries the CUDA event pair for that
                        # microbatch, so reuse them to synchronize hand-offs.
                        flatten_input[idx] = item[batch_idx]
                        forward_event.add(item.transfer_event[batch_idx][0])
                        backward_event.add(item.transfer_event[batch_idx][1])
            except IndexError:
                warnings.warn(f'Batch index {batch_idx} out of range for RoundPipePackedData input, downsizing batch size to {batch_idx}.')
                break

            self.flatten_states.append(flatten_input)
            self.flatten_specs.append(flatten_spec)
            self.forward_events.append(list(forward_event))
            self.backward_events.append(list(backward_event))

        self.num_microbatch: int = len(self.flatten_states)

        if self.num_microbatch == 1:
            self.label_list: List[Any] = [label]
        elif callable(run_config.split_label):
            label_list = run_config.split_label(label, self.num_microbatch)
            assert isinstance(label_list, list) and len(label_list) == self.num_microbatch, \
                   "split_label function must return a list of labels with length equal to num_microbatch."
            self.label_list = label_list
        else:
            label_spec = run_config.split_label
            if label_spec is None:
                label_spec, _ = guess_split_spec(label)
            label_list, _ = split_args_kwargs_into_chunks(
                (label,), {}, self.num_microbatch, (label_spec,), {})
            self.label_list = [lbl_tuple[0] for lbl_tuple in label_list]
        self.loss_list: List[Union[Sequence[torch.Tensor], torch.Tensor]] = [[] for _ in range(self.num_microbatch)]

    def dump(self, run_config: FullRoundPipeRunConfig) -> Any:
        """Merge microbatch outputs according to the provided config.

        When ``merge_output`` is ``False`` a ``RoundPipePackedData`` is
        returned so downstream RoundPipe models can pipeline directly without
        synchronizing to CPU. Otherwise this method blocks until the last CUDA
        event has completed and merges the flattened outputs.

        Args:
            run_config: Effective runtime configuration controlling merging.

        Returns:
            Pytree produced by merging the flattened buffers or the packed
                data wrapper in the same pytree structure when passthrough
                is requested.
        """
        if isinstance(run_config.merge_output, bool) and not run_config.merge_output:
            transfer_events = []
            for i in range(self.num_microbatch):
                forward_events = self.forward_events[i]
                backward_events = self.backward_events[i]
                assert len(forward_events) == 1
                assert len(backward_events) <= 1
                transfer_events.append((forward_events[0], backward_events[0] if len(backward_events) == 1 else torch.cuda.Event()))
            flatten_output = []
            for out_idx in range(len(self.flatten_states[0])):
                batched_out = [self.flatten_states[i][out_idx] for i in range(self.num_microbatch)]
                flatten_output.append(RoundPipePackedData(batched_out, transfer_events))
            return tree_unflatten(flatten_output, self.flatten_specs[0])

        if run_config.output_device != torch.device('cpu'):
            flatten_states_on_device = []
            for flatten_state, (transfer_event,) in zip(self.flatten_states, self.forward_events):
                transfer_event.synchronize()
                flatten_state_on_device = []
                for arg in flatten_state:
                    if isinstance(arg, torch.Tensor):
                        flatten_state_on_device.append(PinnedUpload.apply(arg, run_config.output_device))
                    else:
                        flatten_state_on_device.append(arg)
                flatten_states_on_device.append(flatten_state_on_device)
            self.flatten_states = flatten_states_on_device
        else:
            self.forward_events[-1][0].synchronize()
        # No out of order backward will happen arcoss the sync boundary.
        # Reset the backward scheduler to avoid connecting two unrelated
        # runs into a single backward graph.
        from .scheduler import backward_schedule_simulator
        backward_schedule_simulator.reset()

        if self.num_microbatch == 1:
            return tree_unflatten(self.flatten_states[0], self.flatten_specs[0])

        if callable(run_config.merge_output):
            hidden_states = [tree_unflatten(state, spec) for state, spec in zip(self.flatten_states, self.flatten_specs)]
            return run_config.merge_output(hidden_states)

        if isinstance(run_config.merge_output, bool) or run_config.merge_output is None:
            guessed_spec = []
            for item in self.flatten_states[0]:
                if isinstance(item, torch.Tensor):
                    if item.ndim > 0:
                        guessed_spec.append(TensorChunkSpec(0))
                    else:
                        # Scalar tensors are averaged to approximate per-microbatch reductions.
                        guessed_spec.append(avg_reducer)
                else:
                    guessed_spec.append(None)
            hidden_states = [tree_unflatten(state, spec) for state, spec in zip(self.flatten_states, self.flatten_specs)]
            guessed_spec = tree_unflatten(guessed_spec, self.flatten_specs[0])
            return merge_chunks(hidden_states, guessed_spec)
        else:
            hidden_states = [tree_unflatten(state, spec) for state, spec in zip(self.flatten_states, self.flatten_specs)]
            return merge_chunks(hidden_states, run_config.merge_output)
