from os.path import basename
from typing import Optional, Union
from collections.abc import Iterable
import numpy as np


from .Utils import RLArgument, NDArray
from .DataGetters import DataGetter
from .EOS import EOS
from .RCParams import RCParams


class Simulation():
    """Documentation for Simulation """  # TODO docstring
    sim_path: str
    data_getter: DataGetter
    eos: EOS
    origin: dict[str, float]

    def __init__(self,
                 sim_path: str,
                 data_getter: Optional[DataGetter],
                 eos_path: Optional[str] = None,
                 origin: Optional[dict[str, float]] = None)

    self.sim_path = sim_path
    self.data_getter = data_getter if data_getter is not None else RCParams['default_gettter']
    self.eos = EOS(eos_path if eos_path is not None else RCParams['default_eos_path'])
    self.origin = origin if origin is not None else dict(x=0, y=0, z=0)

    self.sim_name = basename(sim_path)
    self.reflevels = data_getter.get_reflevel()
    self.finest_rl = self.reflevels.max()

    def expand_rl(self, rls: RLArgument) -> NDArray[int]:
        """Expand the "rl" argument to a valid array of refinementlevels"""
        if rls is None:
            return self.reflevels
        if not isinstance(rls, Iterable):
            rls = np.array([rls, ])
        elif ... in rls:
            rls = tuple(rls)
            el_ind = rls.index(...)
            bounds = sorted([rls[el_ind-1], rls[el_ind+1]])
            el_ex = tuple(range(bounds[0], bounds[1]+1))
            rls = np.sort(rls[:el_ind-1] + el_ex + rls[el_ind+2:])
        rls = np.array(rls)
        rls[rls < 0] += self.finest_rl + 1
        return np.sort(rls)

    def get_time(self, it: Union[int, NDArray[int]]) -> Union[float, NDArray[float]]:
        ...

    def get_restart(self, it: Union[int, NDArray[int]]) -> Union[int, NDArray[int]]:
        ...
