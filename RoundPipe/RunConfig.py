from typing import * # type: ignore[reportWildcardImportFromLibrary]

import torch

class RoundPipeRunConfig:
    def __init__(self,
                 requires_grad: Optional[bool] = None,
                 output_device: Optional[torch.device] = None,
                 preserve_rng_state: Optional[bool] = None,
                 num_microbatch: Optional[int] = None,
                 split_input: Union[Tuple[Optional[Tuple], Optional[Dict[str, Any]]],
                                          Callable[[Tuple, Dict[str, Any], int],
                                                   Tuple[List[Tuple], List[Dict[str, Any]]]],
                                    None] = None,
                 merge_output: Union[Any, Callable[[List[Any]], Any], bool, None] = None
                 ):
        self.requires_grad = requires_grad
        self.output_device = output_device
        self.preserve_rng_state = preserve_rng_state
        self.num_microbatch = num_microbatch
        self.split_input = split_input
        self.merge_output = merge_output

class FullRoundPipeRunConfig:
    def __init__(self,
                 function_run_config: RoundPipeRunConfig,
                 model_run_config: RoundPipeRunConfig):
        self.requires_grad \
            = (function_run_config.requires_grad if function_run_config.requires_grad is not None
               else (model_run_config.requires_grad if model_run_config.requires_grad is not None
                     else torch.is_grad_enabled()))
        self.output_device \
            = (function_run_config.output_device if function_run_config.output_device is not None
               else (model_run_config.output_device if model_run_config.output_device is not None
                     else torch.device('cpu')))
        self.preserve_rng_state \
            = (function_run_config.preserve_rng_state if function_run_config.preserve_rng_state is not None
               else (model_run_config.preserve_rng_state if model_run_config.preserve_rng_state is not None
                     else True))
        self.num_microbatch \
            = (function_run_config.num_microbatch if function_run_config.num_microbatch is not None
               else (model_run_config.num_microbatch if model_run_config.num_microbatch is not None
                     else torch.cuda.device_count() + 1))
        self.split_input \
            = (function_run_config.split_input if function_run_config.split_input is not None
               else (model_run_config.split_input if model_run_config.split_input is not None
                     else (None, None)))
        self.merge_output \
            = (function_run_config.merge_output if function_run_config.merge_output is not None
               else model_run_config.merge_output)
