from typing import * # type: ignore[reportWildcardImportFromLibrary]
import importlib

SUPPORTED_MODELS = {
    'Qwen3ForCausalLM': 'RoundPipe.models.qwen3',
}

def wrap_model(model: Any, **roundpipe_kwargs: Any) -> Any:
    model_type = type(model).__name__
    if model_type in SUPPORTED_MODELS:
        module = importlib.import_module(SUPPORTED_MODELS[model_type])
        expected_class = getattr(module, 'EXPECTED_MODEL_CLASS')
        if not isinstance(model, expected_class):
            raise NotImplementedError(f'Model class {type(model)} is not an instance of expected class {expected_class}.')
        wrap_func = getattr(module, 'wrap_model')
        return wrap_func(model, **roundpipe_kwargs)
    else:
        raise NotImplementedError(f'Model type {model_type} does not have a sequential preset.')

__all__ = ['wrap_model']
