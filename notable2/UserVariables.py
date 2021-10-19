import os
import importlib.util
from typing import TYPE_CHECKING, ItemsView, Any

from .EOS import EOS

if TYPE_CHECKING:
    from .Utils import UserVariable


def get_user_variables(path: str, eos: EOS) -> ItemsView[str, dict[str, Any]]:
    name = os.path.basename(path).replace(".py", "")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None:
        raise RuntimeError(f"Could not load {path}")
    uvars = importlib.util.module_from_spec(spec)
    if uvars is None:
        raise RuntimeError(f"Could not load {path}")
    spec.loader.exec_module(uvars)  # type: ignore

    if callable(uvars.user_variables):  # type: ignore
        return uvars.user_variables(eos).items()  # type: ignore
    return uvars.user_variables.items()  # type: ignore
