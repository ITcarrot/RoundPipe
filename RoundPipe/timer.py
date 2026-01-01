"""CUDA event helpers for measuring per-layer forward/backward latency."""

from typing_extensions import *

import torch

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
    """Persistent tracker for smoothed per-layer timings.

    Attributes:
        fwd_timing_events: Per-layer queues of recorded forward events.
        bwd_timing_events: Per-layer queues of recorded backward events.
        fwd_history: Smoothed forward timing history.
        bwd_history: Smoothed backward timing history.
        num_records: Number of timing updates applied so far.
    """

    def __init__(self, num_layers: int):
        """Allocate timing event queues for ``num_layers`` entries.

        Args:
            num_layers: Number of logical layers being timed.
        """
        self.fwd_timing_events: List[List[Tuple[torch.cuda.Event, torch.cuda.Event]]] \
            = [[] for _ in range(num_layers)]
        self.bwd_timing_events: List[List[Tuple[torch.cuda.Event, torch.cuda.Event]]] \
            = [[] for _ in range(num_layers)]
        self.fwd_history: List[float] = [0. for _ in range(num_layers)]
        self.bwd_history: List[float] = [0. for _ in range(num_layers)]
        self.num_records: int = -1

    def time_forward(self, layer_id: int, stream: torch.cuda.Stream) -> LayerTimingContext:
        """Create a timing context for a forward pass.

        Args:
            layer_id: Index of the layer being measured.
            stream: CUDA stream to associate with the events.

        Returns:
            ``LayerTimingContext`` that records start/end events upon use.
        """
        start_event = cast(torch.cuda.Event, torch.cuda.Event(enable_timing = True))
        end_event = cast(torch.cuda.Event, torch.cuda.Event(enable_timing = True))
        self.fwd_timing_events[layer_id].append((start_event, end_event))
        return LayerTimingContext(start_event, end_event, stream)
    
    def time_backward(self, layer_id: int, stream: torch.cuda.Stream) -> LayerTimingContext:
        """Create a timing context for a backward pass.

        Args:
            layer_id: Index of the layer being measured.
            stream: CUDA stream to associate with the events.

        Returns:
            ``LayerTimingContext`` that records start/end events upon use.
        """
        start_event = cast(torch.cuda.Event, torch.cuda.Event(enable_timing = True))
        end_event = cast(torch.cuda.Event, torch.cuda.Event(enable_timing = True))
        self.bwd_timing_events[layer_id].append((start_event, end_event))
        return LayerTimingContext(start_event, end_event, stream)

    def update_workload(self, workload: List[float]) -> bool:
        """Convert recorded events into smoothed timing estimates.

        Args:
            workload: Mutable list updated in-place with averaged timings.

        Returns:
            ``True`` once at least two records exist (ignoring warm-up).
        """
        for idx, (layer_events, layer_history) in enumerate(zip(self.fwd_timing_events, self.fwd_history)):
            new_time = 0.
            for start_event, end_event in layer_events:
                new_time += start_event.elapsed_time(end_event)
            layer_events.clear()
            if self.num_records > -1: # ignore first forward to avoid kernel JIT
                self.fwd_history[idx] = (layer_history * self.num_records + new_time) / (self.num_records + 1)
                workload[idx] = self.fwd_history[idx]
        self.num_records += 1
        return self.num_records > 0
