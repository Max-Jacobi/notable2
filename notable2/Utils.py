from typing import Union, Optional
from collections.abc import Iterable, Mapping
import numpy as np


if np.version.short_version < "1.21":
    from nptyping import NDArray
else:
    NDArray = np.typing.NDArray

RLArgument = Optional[Union[int, Iterable[Union[int, Ellipsis.__class__]]]]

Simulation = ".Simulation.Simulation"


class GridStructure(Mapping):
    """Wrapper around dictionary for the structure of a simulation grid
    Keypatter: variable key -> region -> rl -> (origins, dxs, Ns)
    Can be accessed like this:
      - grid_structure["key/region"] -> dict[rl]
      - grid_structure["key"] -> dict[region][rl]
    """

    def __init__(self, structure_dict: dict):
        self._dict = structure_dict

    def __getitem__(self, string: str) -> Union[dict[str, dict], dict[int, tuple[float]]]:
        keys = string.split('/')
        ret = self._dict
        for kk in keys:
            ret = ret[kk]
        return ret

    def __iter__(self):
        return self._dict.__iter__()
