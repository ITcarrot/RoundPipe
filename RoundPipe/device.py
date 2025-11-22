from beartype.typing import * # type: ignore[reportWildcardImport]
from enum import Enum
import threading

import torch

from .batch import Batch
from .run import run_forward, run_backward, RoundPipeRunContext

class JobType(Enum):
    FORWARD = 1
    BACKWARD = 2

class DeviceManager:
    def __init__(self, id: int, device: torch.device):
        self.id = id
        self.device = device
        
        self.upstream: torch.cuda.Stream = torch.cuda.Stream(device) # type: ignore[reportAttributeAccessIssue]
        self.compute_stream = torch.cuda.default_stream(self.device)
        self.downstream: torch.cuda.Stream = torch.cuda.Stream(device) # type: ignore[reportAttributeAccessIssue]

        self.is_idle = threading.Semaphore(1)
        self.job_arrived = threading.Semaphore(0)
        self.cur_job: Optional[Tuple[JobType, int, Optional[Batch], List[RoundPipeRunContext]]] = None
        self.controller_thread = threading.Thread(target=self.controller, daemon=True, name=f'DeviceController-{id}')
        self.controller_thread.start()

    def controller(self) -> None:
        while True:
            self.job_arrived.acquire()
            assert self.cur_job is not None
            job_type, layer_group_id, batch, run_context = self.cur_job
            if job_type == JobType.FORWARD:
                assert batch is not None
                for batch_context in run_context:
                    run_forward(layer_group_id, batch, batch_context, self)
            elif job_type == JobType.BACKWARD:
                for batch_context in run_context:
                    run_backward(layer_group_id, batch_context, self)
            self.is_idle.release()

    def launch_forward(self, layer_group_id: int, batch: Batch,
                       run_context: List[RoundPipeRunContext]) -> None:
        self.is_idle.acquire()
        self.cur_job = (JobType.FORWARD, layer_group_id, batch, run_context)
        self.job_arrived.release()
    
    def launch_backward(self, layer_group_id: int,
                        run_context: List[RoundPipeRunContext]) -> None:
        self.is_idle.acquire()
        self.cur_job = (JobType.BACKWARD, layer_group_id, None, run_context)
        self.job_arrived.release()

device_list: List[DeviceManager] = []
cur_device = 0

for i in range(torch.cuda.device_count()):
    device = DeviceManager(i, torch.device(f"cuda:{i}"))
    device_list.append(device)

def get_next_device() -> DeviceManager:
    global cur_device
    device = device_list[cur_device]
    cur_device = (cur_device + 1) % len(device_list)
    return device
