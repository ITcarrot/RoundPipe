from beartype.typing import * # type: ignore[reportWildcardImportFromLibrary]

import torch

class LayerTimingContext:
    def __init__(self,
                 start_event: torch.cuda.Event, end_event: torch.cuda.Event,
                 stream: Optional[torch.cuda.Stream]):
        self.start_event = start_event
        self.end_event = end_event
        self.stream = stream
        
    def __enter__(self):
        self.start_event.record(self.stream) # type: ignore[reportArgumentType]
    
    def __exit__(self, *args):
        self.end_event.record(self.stream) # type: ignore[reportArgumentType]

class ModelTimer:
    def __init__(self, num_layers: int):
        self.fwd_timing_events: List[List[Tuple[torch.cuda.Event, torch.cuda.Event]]] \
            = [[] for _ in range(num_layers)]
        self.bwd_timing_events: List[List[Tuple[torch.cuda.Event, torch.cuda.Event]]] \
            = [[] for _ in range(num_layers)]
        self.fwd_history = [0. for _ in range(num_layers)]
        self.bwd_history = [0. for _ in range(num_layers)]
        self.num_records = -1

    def time_forward(self, layer_id: int, stream: Optional[torch.cuda.Stream] = None) -> LayerTimingContext:
        start_event: torch.cuda.Event = torch.cuda.Event(enable_timing = True)  # type: ignore[reportAssignmentType]
        end_event: torch.cuda.Event = torch.cuda.Event(enable_timing = True)  # type: ignore[reportAssignmentType]
        self.fwd_timing_events[layer_id].append((start_event, end_event))
        return LayerTimingContext(start_event, end_event, stream)
    
    def time_backward(self, layer_id: int, stream: Optional[torch.cuda.Stream] = None) -> LayerTimingContext:
        start_event: torch.cuda.Event = torch.cuda.Event(enable_timing = True)  # type: ignore[reportAssignmentType]
        end_event: torch.cuda.Event = torch.cuda.Event(enable_timing = True)  # type: ignore[reportAssignmentType]
        self.bwd_timing_events[layer_id].append((start_event, end_event))
        return LayerTimingContext(start_event, end_event, stream)

    def update_workload(self, workload: List[float]) -> bool:
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
