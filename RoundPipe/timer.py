"""CUDA event helpers for measuring per-layer forward/backward latency."""

from typing_extensions import *
import sys

import torch

from .utils import get_model_size

class LayerTimingContext:
    """Context manager that records CUDA events around a code block.

    Attributes:
        start_event: CUDA event recorded on entry.
        end_event: CUDA event recorded on exit.
        stream: Stream used for event recording.
    """

    def __init__(self,
                 start_event: torch.cuda.Event, end_event: torch.cuda.Event,
                 stream: torch.cuda.Stream):
        """Store events/stream for later use.

        Args:
            start_event: Event recorded at context entry.
            end_event: Event recorded at context exit.
            stream: CUDA stream on which to record events.
        """
        self.start_event: torch.cuda.Event = start_event
        self.end_event: torch.cuda.Event = end_event
        self.stream: torch.cuda.Stream = stream
        
    def __enter__(self):
        """Record ``start_event`` on the configured stream."""
        self.start_event.record(self.stream)
    
    def __exit__(self, *args):
        """Record ``end_event`` when the context exits."""
        self.end_event.record(self.stream)

class ModelTimer:
    """Tracks per-layer forward/recompute/backward timing using CUDA events.
    Before having any records, the timer uses model size-based estimates.
    After one warm-up run, the timer updates its estimates using an exponential
    moving average from 0.
    
    Attributes:
        SMOOTH_RATE: Smoothing factor for time estimates.
        BACKWARD_MULTIPLIER: Initial multiplier to estimate backward time
            from forward time.
        VERBOSE: Whether to print timing updates.
        n_layers: Number of layers being timed.
        stage: 0 if no events recorded yet, 1 if first result dropped,
            2 if time-based estimate has been computed.
        scale: Scaling factors for each type. Since moving average starts
            from 0, this tracks the ineffective weight of averages.
        estimate: Smoothed time estimates for each layer and type.
        fwd_events: Recorded forward/recompute events for each layer
            from current iteration.
        bwd_events: Recorded backward events for each layer
            from current iteration.
        pending_fwd: Forward/recompute events from previous iteration.
        pending_bwd: Backward events from previous iteration.
    """
    SMOOTH_RATE: float = 0.9
    BACKWARD_MULTIPLIER: float = 2.0
    VERBOSE: bool = False
    def __init__(self, layers: List[torch.nn.Module]):
        self.n_layers: int = len(layers)
        self.stage: Dict[Literal['fwd', 're', 'bwd'],  Literal[0, 1, 2]] = {
            'fwd': 0, 're': 0, 'bwd': 0,
        }
        self.scale: Dict[Literal['fwd', 're', 'bwd'], float] = {
            'fwd': 0.0, 're': 0.0, 'bwd': 0.0,
        }
        self.estimate: Dict[Literal['fwd', 're', 'bwd'], List[float]] = {
            'fwd': [float(get_model_size(layer)) for layer in layers],
            're': [float(get_model_size(layer)) for layer in layers],
            'bwd': [float(get_model_size(layer) * self.BACKWARD_MULTIPLIER)
                    for layer in layers],
        }
        self.fwd_events: Dict[Literal['fwd', 're'],
                              List[List[Tuple[torch.cuda.Event, torch.cuda.Event]]]] = {
            'fwd': [[] for _ in layers],
            're': [[] for _ in layers],
        }
        self.bwd_events: Dict[range, List[Tuple[torch.cuda.Event, torch.cuda.Event]]] = {}
        self.pending_fwd: Dict[Literal['fwd', 're'],
                                  List[List[Tuple[torch.cuda.Event, torch.cuda.Event]]]] = {
            'fwd': [[] for _ in layers],
            're': [[] for _ in layers],
        }
        self.pending_bwd: Dict[range, List[Tuple[torch.cuda.Event, torch.cuda.Event]]] = {}

    def time_fwd(self, action: Literal['fwd', 're'], layer_idx: int,
                 stream: torch.cuda.Stream) -> LayerTimingContext:
        """Create a context manager to time a forward/recompute layer.

        Args:
            action: Either 'fwd' or 're' to indicate forward or recompute.
            layer_idx: Index of the layer being timed.
            stream: CUDA stream on which to record events.

        Returns:
            A LayerTimingContext that records events on entry/exit.
        """
        start_event = cast(torch.cuda.Event, torch.cuda.Event(enable_timing=True))
        end_event = cast(torch.cuda.Event, torch.cuda.Event(enable_timing=True))
        self.fwd_events[action][layer_idx].append((start_event, end_event))
        return LayerTimingContext(start_event, end_event, stream)

    def time_bwd(self, layer_ids: range, stream: torch.cuda.Stream
                 ) -> LayerTimingContext:
        """Create a context manager to time a backward layer.

        Args:
            layer_ids: Range of layer indices being timed.
            stream: CUDA stream on which to record events.

        Returns:
            A LayerTimingContext that records events on entry/exit.
        """
        start_event = cast(torch.cuda.Event, torch.cuda.Event(enable_timing=True))
        end_event = cast(torch.cuda.Event, torch.cuda.Event(enable_timing=True))
        self.bwd_events.setdefault(layer_ids, []).append((start_event, end_event))
        return LayerTimingContext(start_event, end_event, stream)

    def update_times(self):
        """Update time estimates based on recorded events.""" 
        sums = {
            'fwd': [0.0] * self.n_layers,
            're': [0.0] * self.n_layers,
            'bwd': [0.0] * self.n_layers,
        }
        cnts = {
            'fwd': [0] * self.n_layers,
            're': [0] * self.n_layers,
            'bwd': [0] * self.n_layers,
        }
        for action in self.pending_fwd.keys():
            for layer_idx, events in enumerate(self.pending_fwd[action]):
                cnts[action][layer_idx] = len(events)
                for start_event, end_event in events:
                    end_event.synchronize()
                    elapsed = start_event.elapsed_time(end_event)
                    sums[action][layer_idx] += elapsed
        assert all(cnts['fwd'][i] == cnts['fwd'][0] for i in range(self.n_layers)), \
            "Mismatched forward event counts across layers"
        assert all(cnts['re'][i] == cnts['re'][0] for i in range(self.n_layers)), \
            "Mismatched recompute event counts across layers"
        # Backward timing proportional to recompute times
        for layer_ids, events in self.pending_bwd.items():
            scale_sum = sum(sums['re'][i] for i in layer_ids) + sys.float_info.epsilon
            block_sum = 0.0
            for start_event, end_event in events:
                end_event.synchronize()
                elapsed = start_event.elapsed_time(end_event)
                block_sum += elapsed
            for i in layer_ids:
                portion = sums['re'][i] / scale_sum * block_sum
                sums['bwd'][i] += portion
                cnts['bwd'][i] += len(events)
        assert all(cnts['bwd'][i] == cnts['bwd'][0] for i in range(self.n_layers)), \
            "Mismatched backward event counts across layers"
        # Update estimates with smoothed averages
        for action in self.estimate.keys():
            if cnts[action][0] > 0:
                if self.stage[action] == 0:
                    self.stage[action] = 1
                    continue  # Drop first result to avoid startup noise
                if self.stage[action] == 1:
                    self.stage[action] = 2
                    self.scale[action] = 1.0
                    for layer_idx in range(self.n_layers):
                        self.estimate[action][layer_idx] = 0.0

                self.scale[action] *= self.SMOOTH_RATE
                for layer_idx in range(self.n_layers):
                    avg_time = sums[action][layer_idx] / cnts[action][layer_idx]
                    self.estimate[action][layer_idx] \
                        = self.SMOOTH_RATE * self.estimate[action][layer_idx] \
                        + (1 - self.SMOOTH_RATE) * avg_time
        if self.VERBOSE:
            for action in self.estimate.keys():
                if cnts[action][0] > 0:
                    for layer_idx in range(self.n_layers):
                        print(f"Layer {layer_idx} {action}  "
                              f"new record: {sums[action][layer_idx] / cnts[action][layer_idx]:.3f} ms  "
                              f"new estimate: {self.estimate[action][layer_idx] / (1.0 - self.scale[action]):.3f} ms")
        # Move current events to pending and clear current
        self.pending_fwd = self.fwd_events
        self.pending_bwd = self.bwd_events
        self.fwd_events = {
            'fwd': [[] for _ in range(self.n_layers)],
            're': [[] for _ in range(self.n_layers)],
        }
        self.bwd_events = {}
