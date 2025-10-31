from typing import * # type: ignore[reportWildcardImportFromLibrary]

SUPPORTED_MODELS = {
    'function': 'RoundPipe.models.function.wrap_function',
    'Qwen3ForCausalLM': 'RoundPipe.models.qwen3.wrap_Qwen3ForCausalLM',
}

def wrap_model(model: Any, **roundpipe_kwargs: Any) -> Any:
    model_type = type(model).__name__
    if model_type in SUPPORTED_MODELS:
        module_path, func_name = SUPPORTED_MODELS[model_type].rsplit('.', 1)
        module = __import__(module_path, fromlist=[func_name])
        wrap_func = getattr(module, func_name)
        return wrap_func(model, **roundpipe_kwargs)
    else:
        raise NotImplementedError(f'Model type {model_type} does not have a sequential preset.')

__all__ = ['wrap_model']
