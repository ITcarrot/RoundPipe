from beartype.typing import * # pyright: ignore[reportWildcardImportFromLibrary]

import itertools

import pytest
from torch import nn

from RoundPipe import RoundPipe

@pytest.mark.parametrize("model_class, designate",
                         itertools.product([RoundPipe], [True, False]))
def test_RoundPipe_attribute_shim(model_class, designate):
    original_model = nn.Linear(10, 5)
    if designate:
        wrapped_model = model_class(nn.Linear(10, 5))
        wrapped_model.set_original_model(original_model)
    else:
        wrapped_model = model_class(original_model)

    obj = {'dict': 123}
    wrapped_model.var1 = 42
    wrapped_model.var2 = obj
    assert original_model.var1 == 42
    assert original_model.var2 is obj
    assert wrapped_model.var1 == 42
    assert wrapped_model.var2 is obj

    del wrapped_model.var1
    del wrapped_model.var2
    assert not hasattr(original_model, 'var1')
    assert not hasattr(original_model, 'var2')
    assert not hasattr(wrapped_model, 'var1')
    assert not hasattr(wrapped_model, 'var2')
