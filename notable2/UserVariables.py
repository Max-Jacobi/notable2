import os
import importlib.util
from typing import TYPE_CHECKING, ItemsView, Any

if TYPE_CHECKING:
    from .Utils import UserVariable


def get_user_variables(path: str) -> ItemsView[str, dict[str, Any]]:
    name = os.path.basename(path).replace(".py", "")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None:
        raise RuntimeError(f"Could not load {path}")
    uvars = importlib.util.module_from_spec(spec)
    if uvars is None:
        raise RuntimeError(f"Could not load {path}")
    spec.loader.exec_module(uvars)  # type: ignore

    return uvars.user_variables.items()  # type: ignore
