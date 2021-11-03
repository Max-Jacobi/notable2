import os
import importlib.util
from typing import TYPE_CHECKING, ItemsView, Any

from .EOS import EOS

if TYPE_CHECKING:
    from .Utils import PostProcVariable


def get_pp_variables(path: str, eos: EOS) -> ItemsView[str, dict[str, Any]]:
    name = os.path.basename(path).replace(".py", "")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None:
        raise RuntimeError(f"Could not load {path}")
    ppvars = importlib.util.module_from_spec(spec)
    if ppvars is None:
        raise RuntimeError(f"Could not load {path}")
    spec.loader.exec_module(ppvars)  # type: ignore

    if callable(ppvars.pp_variables):  # type: ignore
        return ppvars.pp_variables(eos).items()  # type: ignore
    return ppvars.pp_variables.items()  # type: ignore
