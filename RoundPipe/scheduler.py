from typing import * # type: ignore[reportWildcardImportFromLibrary]

if TYPE_CHECKING:
    from RoundPipe.RoundPipe import RoundPipe

class ModelExecutePlan:
    def __init__(self, model: 'RoundPipe', fuse_fwd_bwd: bool) -> None:
        self.fwd_plan: List[range] = []
        self.bwd_plan: List[range] = []

        if fuse_fwd_bwd:
            self.fwd_plan = [range(i, i + 1) for i in range(model.num_layers - 1)]
            self.bwd_plan = [range(i, i + 1) for i in reversed(range(model.num_layers))]
        else:
            self.fwd_plan = [range(i, i + 1) for i in range(model.num_layers)]
            self.bwd_plan = list(reversed(self.fwd_plan))

    def backward_need_input(self, layer_id: int) -> bool:
        return any(r[0] == layer_id for r in self.bwd_plan)
