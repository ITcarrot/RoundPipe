from typing import * # type: ignore[reportWildcardImportFromLibrary]
import itertools

import pytest
import torch
from torch.distributed.pipelining.microbatch import TensorChunkSpec, _Replicate, sum_reducer

from RoundPipe.batch import Batch, RoundPipePackedData
from RoundPipe.RunConfig import RoundPipeRunConfig, FullRoundPipeRunConfig

@pytest.mark.parametrize("num_microbatch, merge_output", itertools.product([2, 3, 4], [True, None]))
def test_io_auto(num_microbatch, merge_output):
    events: List[List[torch.cuda.Event]] = [[torch.cuda.Event() for _ in range(num_microbatch)] for _ in range(3)] # type: ignore[assignment]
    packed1 = RoundPipePackedData([torch.randn(12, 14) for i in range(num_microbatch)], [(events[0][i], events[0][i]) for i in range(num_microbatch)])
    packed2 = RoundPipePackedData([torch.randn(12, 16) for i in range(num_microbatch)], [(events[1][i], events[1][i]) for i in range(num_microbatch)])
    packed3 = RoundPipePackedData([torch.randn(12, 18) for i in range(num_microbatch)], [(events[2][i], events[2][i]) for i in range(num_microbatch)])
    args = (torch.tensor(233), torch.randn(12, 3, 4), packed1,
            (torch.randn(12, 6), torch.randn(12, 7), 'string in tuple'),
            [torch.randn(12, 6), torch.randn(12, 7), 'string in list'],
            {'a': torch.randn(12, 8), 'packed': packed2, 'string': 'string in dict'},
            'non-tensor argument', 42)
    kwargs = {'x': torch.randn(12, 10), 'y': [torch.randn(12, 11), 'string in list', packed2],
              'z': {'inner_tensor': torch.randn(12, 12), 'inner_string': 'inner string'},
              'non_tensor': 3.14, 'tuple_arg': (torch.randn(12, 5, 6), 'string in tuple'),
              'packed_arg': packed3}

    auto_config = FullRoundPipeRunConfig(RoundPipeRunConfig(num_microbatch=num_microbatch, merge_output=merge_output), RoundPipeRunConfig())
    batch = Batch(args, kwargs, auto_config)
    for batch_idx in range(num_microbatch):
        fwd_events, bwd_events = batch.transfer_events[batch_idx]
        for id in range(3):
            assert events[id][batch_idx] in fwd_events
            assert events[id][batch_idx] in bwd_events
    args_reconstruct, kwargs_reconstruct = batch.dump(auto_config)
    
    assert args[0] * num_microbatch == args_reconstruct[0]
    assert torch.allclose(args[1], args_reconstruct[1])
    assert torch.allclose(torch.cat(args[2]), args_reconstruct[2])
    assert torch.allclose(args[3][0], args_reconstruct[3][0])
    assert torch.allclose(args[3][1], args_reconstruct[3][1])
    assert args[3][2] == args_reconstruct[3][2]
    assert torch.allclose(args[4][0], args_reconstruct[4][0])
    assert torch.allclose(args[4][1], args_reconstruct[4][1])
    assert args[4][2] == args_reconstruct[4][2]
    assert torch.allclose(args[5]['a'], args_reconstruct[5]['a'])
    assert torch.allclose(torch.cat(args[5]['packed']), args_reconstruct[5]['packed'])
    assert args[5]['string'] == args_reconstruct[5]['string']
    assert args[6] == args_reconstruct[6]
    assert args[7] == args_reconstruct[7]
    
    assert torch.allclose(kwargs['x'], kwargs_reconstruct['x'])
    assert torch.allclose(kwargs['y'][0], kwargs_reconstruct['y'][0])
    assert kwargs['y'][1] == kwargs_reconstruct['y'][1]
    assert torch.allclose(torch.cat(kwargs['y'][2]), kwargs_reconstruct['y'][2])
    assert torch.allclose(kwargs['z']['inner_tensor'], kwargs_reconstruct['z']['inner_tensor'])
    assert kwargs['z']['inner_string'] == kwargs_reconstruct['z']['inner_string']
    assert kwargs['non_tensor'] == kwargs_reconstruct['non_tensor']
    assert torch.allclose(kwargs['tuple_arg'][0], kwargs_reconstruct['tuple_arg'][0])
    assert kwargs['tuple_arg'][1] == kwargs_reconstruct['tuple_arg'][1]
    assert torch.allclose(torch.cat(kwargs['packed_arg']), kwargs_reconstruct['packed_arg'])

@pytest.mark.parametrize("num_microbatch", [2, 3, 4])
def test_io_spec(num_microbatch):
    events: List[torch.cuda.Event] = [torch.cuda.Event() for _ in range(num_microbatch)] # type: ignore[assignment]
    packed = RoundPipePackedData([torch.randn(12, 14) for i in range(num_microbatch)], [(events[i], events[i]) for i in range(num_microbatch)])
    args = (torch.tensor(233), torch.randn(3, 12, 4))
    kwargs = {'x': torch.randn(10, 12), 'packed_arg': packed, 'non_tensor': 'hello'}
    args_spec = (_Replicate, TensorChunkSpec(1))
    kwargs_spec = {'x': TensorChunkSpec(0), 'packed_arg': TensorChunkSpec(0), 'non_tensor': _Replicate}
    merge_spec = ((sum_reducer, TensorChunkSpec(1)), {'x': TensorChunkSpec(0), 'packed_arg': TensorChunkSpec(0), 'non_tensor': None})
    
    spec_config = FullRoundPipeRunConfig(RoundPipeRunConfig(num_microbatch=num_microbatch,
                                                            split_input=(args_spec, kwargs_spec),
                                                            merge_output=merge_spec), RoundPipeRunConfig())
    batch = Batch(args, kwargs, spec_config)
    for batch_idx in range(num_microbatch):
        fwd_events, bwd_events = batch.transfer_events[batch_idx]
        assert events[batch_idx] in fwd_events
        assert events[batch_idx] in bwd_events
    args_reconstruct, kwargs_reconstruct = batch.dump(spec_config)

    assert args[0] * num_microbatch == args_reconstruct[0]
    assert torch.allclose(args[1], args_reconstruct[1])
    assert torch.allclose(torch.cat(kwargs['packed_arg']), kwargs_reconstruct['packed_arg'])
    assert torch.allclose(kwargs['x'], kwargs_reconstruct['x'])
    assert kwargs['non_tensor'] == kwargs_reconstruct['non_tensor']

@pytest.mark.parametrize("num_microbatch", [1, 2, 3, 4])
def test_out_packed(num_microbatch):
    events: List[torch.cuda.Event] = [torch.cuda.Event() for _ in range(num_microbatch)] # type: ignore[assignment]
    packed = RoundPipePackedData([torch.randn(12, i) for i in range(num_microbatch)], [(events[i], events[i]) for i in range(num_microbatch)])
    args = (torch.tensor(233), torch.randn(12, 3, 4))
    kwargs = {'x': torch.randn(12, 10), 'packed_arg': packed, 'non_tensor': 'hello'}
    
    spec_config = FullRoundPipeRunConfig(RoundPipeRunConfig(num_microbatch=num_microbatch,
                                                            merge_output=False), RoundPipeRunConfig())
    batch = Batch(args, kwargs, spec_config)
    for batch_idx in range(num_microbatch):
        fwd_events, bwd_events = batch.transfer_events[batch_idx]
        assert events[batch_idx] in fwd_events
        assert events[batch_idx] in bwd_events
    args_reconstruct, kwargs_reconstruct = batch.dump(spec_config)

    if num_microbatch == 1:
        assert args[0] == args_reconstruct[0]
        assert torch.allclose(args[1], args_reconstruct[1])
        assert torch.allclose(kwargs['x'], kwargs_reconstruct['x'])
        assert torch.allclose(torch.cat(kwargs['packed_arg']), kwargs_reconstruct['packed_arg'])  # empty tensor
        assert kwargs['non_tensor'] == kwargs_reconstruct['non_tensor']
    else:
        expected_transfer_event = [(events[i], events[i]) for i in range(num_microbatch)]
        assert args_reconstruct[0].transfer_event == expected_transfer_event
        assert args_reconstruct[1].transfer_event == expected_transfer_event
        assert kwargs_reconstruct['x'].transfer_event == expected_transfer_event
        assert kwargs_reconstruct['packed_arg'].transfer_event == expected_transfer_event
        assert kwargs_reconstruct['non_tensor'].transfer_event == expected_transfer_event
        args1_chunk = torch.chunk(args[1], num_microbatch)
        kwargs_x_chunk = torch.chunk(kwargs['x'], num_microbatch)
        for batch_idx in range(num_microbatch):
            assert args_reconstruct[0][batch_idx] == args[0]
            assert torch.allclose(args_reconstruct[1][batch_idx], args1_chunk[batch_idx])
            assert torch.allclose(kwargs_reconstruct['x'][batch_idx], kwargs_x_chunk[batch_idx])
            assert torch.allclose(kwargs_reconstruct['packed_arg'][batch_idx], packed[batch_idx])
            assert kwargs_reconstruct['non_tensor'][batch_idx] == kwargs['non_tensor']
