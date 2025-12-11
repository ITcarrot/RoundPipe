"""Runtime configuration objects shared across RoundPipe components."""

from beartype.typing import * # pyright: ignore[reportWildcardImportFromLibrary]
from beartype import beartype

import torch

@beartype
class RoundPipeRunConfig:
    """Shallow user-facing configuration applied per forward/train call."""
    def __init__(self,
                 requires_grad: Optional[bool] = None,
                 output_device: Optional[torch.device] = None,
                 preserve_rng_state: Optional[bool] = None,
                 num_microbatch: Optional[int] = None,
                 split_input: Union[Tuple[Optional[Tuple], Optional[Dict[str, Any]]],
                                    Callable[[Tuple, Dict[str, Any], int],
                                              Tuple[List[Tuple], List[Dict[str, Any]]]],
                                    None] = None,
                 split_label: Union[Any, Callable[[Any, int], List[Any]], None] = None,
                 merge_output: Union[Any, Callable[[List[Any]], Any], bool, None] = None
                 ) -> None:
        """
        Configuration for running RoundPipe models.
        User may specify model-level configuration when initializing RoundPipe, and/or function-level configuration when calling forward().
        
        Parameters:
            requires_grad: Whether to enable gradient computation. If None, defaults to the global setting.
            output_device: The device where the output tensors will be placed. If None, defaults to CPU.
            preserve_rng_state: Whether to preserve the random number generator state. If None, defaults to True.
            num_microbatch: The number of microbatches to split the input into. If None, defaults to the number of available CUDA devices plus one.
            split_input (-): Specifies how to split input arguments into microbatches. If None, defaults to automatic splitting.
            split_label (-): Specifies how to split labels into microbatches. If None, defaults to automatic splitting.
            merge_output (-): Specifies how to merge output microbatches back into a single output. If None, defaults to automatic merging.

        """
        self.requires_grad = requires_grad
        self.output_device = output_device
        self.preserve_rng_state = preserve_rng_state
        self.num_microbatch = num_microbatch
        self.split_input = split_input
        self.split_label = split_label
        self.merge_output = merge_output

    def __str__(self) -> str:
        string = f'RoundPipeRunConfig('
        if self.requires_grad is not None:
            string += f'requires_grad={self.requires_grad}, '
        if self.output_device is not None:
            string += f'output_device={self.output_device}, '
        if self.preserve_rng_state is not None:
            string += f'preserve_rng_state={self.preserve_rng_state}, '
        if self.num_microbatch is not None:
            string += f'num_microbatch={self.num_microbatch}, '
        if self.split_input is not None:
            string += f'split_input={self.split_input}, '
        if self.split_label is not None:
            string += f'split_label={self.split_label}, '
        if self.merge_output is not None:
            string += f'merge_output={self.merge_output}, '
        return string[:-2] + ')' if string.endswith(', ') else string + ')'

    def __repr__(self) -> str:
        return self.__str__()

class FullRoundPipeRunConfig:
    """Resolved configuration combining model defaults and call overrides.

    See ``RoundPipeRunConfig`` for parameter details.
    """

    def __init__(self,
                 function_run_config: RoundPipeRunConfig,
                 model_run_config: RoundPipeRunConfig):
        """Merge a per-call config with the model's baseline config.

        Args:
            function_run_config: Configuration passed to ``RoundPipe.forward``.
            model_run_config: Baseline configuration stored on the model.
        """
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
        self.split_label \
            = (function_run_config.split_label if function_run_config.split_label is not None
               else model_run_config.split_label)
        self.merge_output \
            = (function_run_config.merge_output if function_run_config.merge_output is not None
               else model_run_config.merge_output)
