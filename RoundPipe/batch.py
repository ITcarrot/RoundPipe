from typing import * # type: ignore[reportWildcardImportFromLibrary]
import warnings

import torch
from torch.distributed.pipelining.microbatch import TensorChunkSpec, _Replicate, split_args_kwargs_into_chunks, merge_chunks, sum_reducer
from torch.utils._pytree import tree_flatten, tree_unflatten

from RoundPipe.transfer import PinnedUpload, RegisterBackwardEvent

if TYPE_CHECKING:
    from torch.utils._pytree import TreeSpec
    from RoundPipe.RunConfig import FullRoundPipeRunConfig

class RoundPipePackedData(list):
    def __init__(self, data: List[Any],
                 transfer_event: List[Tuple[torch.cuda.Event, torch.cuda.Event]]) -> None:
        super().__init__(data)
        self.transfer_event = transfer_event

def guess_split_spec(data: Any, expected_batchsize: Optional[int] = None) -> Tuple[Any, Optional[int]]:
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

class Batch:
    def __init__(self, args: Tuple, kwargs: Dict[str, Any],
                 run_config: 'FullRoundPipeRunConfig') -> None:
        if run_config.num_microbatch == 1:
            args_list, kwargs_list = [args], [kwargs]
        elif callable(run_config.split_input):
            args_list, kwargs_list = run_config.split_input(args, kwargs, run_config.num_microbatch)
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
        self.flatten_specs: List['TreeSpec'] = []
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
                    if item.requires_grad:
                        flatten_input[idx] = RegisterBackwardEvent.apply(item, cpu_tensor_backward_event)
                        backward_event.add(cpu_tensor_backward_event)
            try:
                for idx, item in enumerate(flatten_input):
                    if isinstance(item, RoundPipePackedData):
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

        self.num_microbatch = len(self.flatten_states)

    def dump(self, run_config: 'FullRoundPipeRunConfig') -> Any:
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
        from RoundPipe.scheduler import backward_schedule_simulator
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
                        guessed_spec.append(sum_reducer)
                else:
                    guessed_spec.append(None)
            hidden_states = [tree_unflatten(state, spec) for state, spec in zip(self.flatten_states, self.flatten_specs)]
            guessed_spec = tree_unflatten(guessed_spec, self.flatten_specs[0])
            return merge_chunks(hidden_states, guessed_spec)
        else:
            hidden_states = [tree_unflatten(state, spec) for state, spec in zip(self.flatten_states, self.flatten_specs)]
            return merge_chunks(hidden_states, run_config.merge_output)
