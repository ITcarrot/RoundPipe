import torch.nn as nn

def get_model_size(model: nn.Module, recurse: bool = True) -> int:
    param_size = sum(p.numel() * p.element_size() for p in model.parameters(recurse))
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers(recurse))
    return param_size + buffer_size
