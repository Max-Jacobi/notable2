from os.path import basename
from typing import Optional, Union, Type, TYPE_CHECKING, Tuple
from collections.abc import Iterable
from functools import reduce
import numpy as np


from .DataHandlers import DataHandler
from .EOS import EOS, TabulateEOS
from .RCParams import rcParams
from .Variable import GridVariable, TimeSeriesVariable, UGridVariable, UTimeSeriesVariable, VariableError
from .Utils import IterationError

if TYPE_CHECKING:
    from .Utils import RLArgument, NDArray, Variable


class Simulation():
    """Documentation for Simulation """  # TODO docstring
    sim_path: str
    sim_name: str
    data_handler: DataHandler
    eos: EOS
    user_variables: dict[str, "Variable"]

    def __init__(self,
                 sim_path: str,
                 data_handler: Optional[Type[DataHandler]] = None,
                 eos_path: Optional[str] = None,
                 offset: Optional[dict[str, float]] = None):
        self.sim_path = sim_path
        self.sim_name = basename(sim_path)

        self.data_handler = data_handler(self) if data_handler is not None \
            else rcParams.default_getter(self)
        if eos_path == 'ideal':
            ...
        else:
            self.eos = TabulateEOS(eos_path if eos_path is not None else rcParams.default_eos_path)
        self._offset = offset if offset is not None else dict(x=0, y=0, z=0)

        (self._its, self._times, self._restarts), self._structure, self._its_lookup \
            = self.data_handler.get_structure()
        self.rls = np.array(list(self._structure[0].keys()))
        self.finest_rl = self.rls.max()
        self.user_variables = {}  # TODO

    def __repr__(self):
        return f"Einstein Toolkit simulation {self.sim_name}"

    def __str__(self):
        return self.sim_name

    def expand_rl(self, rls: "RLArgument") -> "NDArray[int]":
        """Expand the "rl" argument to a valid array of refinementlevels"""
        if rls is None:
            return self.rls
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

    def get_time(self, it: Union[int, "NDArray[int]"]) -> Union[float, "NDArray[float]"]:
        """Get the time for iteration(s) it"""
        # python 3.10 match(it)
        if isinstance(it, np.ndarray):
            if any(ii not in self._its for ii in it):
                it_er = self._its[np.array([ii not in self._its for ii in it])]
                raise IterationError(f"Iterations {it_er} not all in {self.sim_name}")
            return self._times[self._its.searchsorted(it)]
        if it not in self._its:
            raise IterationError(f"Iteration {it} not all in {self.sim_name}")
        return self._times[np.argwhere(self._its == it)][0][0]

    def get_restart(self, it: Union[int, "NDArray[int]"]) -> Union[int, "NDArray[int]"]:
        """Get the restart number for iteration(s) it"""
        if isinstance(it, np.ndarray):
            if any(ii not in self._its for ii in it):
                it_er = self._its[np.array([ii not in self._its for ii in it])]
                raise IterationError(f"Iterations {it_er} not all in {self.sim_name}")
            return self._times[self._its.searchsorted(it)]
        if it not in self._its:
            raise IterationError(f"Iteration {it} not all in {self.sim_name}")
        return self._restarts[np.argwhere(self._its)]

    def get_it(self, time: Union[float, "NDArray[float]"]) -> Union[int, "NDArray[int]"]:
        """Get the smallest itteration(s) with time < the given time"""
        return self._its[self._times.searchsorted(time)]

    def get_variable(self, key: str) -> "Variable":
        """Return Variable for key"""
        for var in [TimeSeriesVariable, GridVariable]:
            try:
                return var(key, self)
            except KeyError:
                continue
        else:
            if key in self.user_variables:
                return self.user_variables[key]
            raise VariableError(f"Could not find key {key} in {self}.")

    def lookup_it(self,
                  var: "Variable",
                  region: str,
                  time: Optional[Union[Tuple[float], float]] = None,
                  it: Optional[Union[Tuple[int], int]] = None) -> Union[int, "NDArray[int]"]:
        """Returns the available iterations for a given Variable
           If neither it or time is given the first available it is returned.
           If it is given a ValueError is raised if it is not available.
           If time is given the closest iteration to that time is returned.
           If a tuple of times or its is given an array of its is returned which are bounded by the touple"""
        if isinstance(var, (UGridVariable, UTimeSeriesVariable)):
            try:
                its = [self._its_lookup[dvar.key][region] for dvar in var.dependencies]
            except VariableError as e:
                raise VariableError(f"Could not get dependencies for UVariable {var}") from e

            its = reduce(np.intersect1d, its)

            if len(its) == 0:
                raise IterationError(f"Could not find common iterations for UVariable {var}")

        elif isinstance(var, GridVariable):
            its = self._its_lookup[var.key][region]

        elif isinstance(var, TimeSeriesVariable):
            its = var.get_data().its

        else:
            raise ValueError

        if it is None and time is None:
            return its[0]
        if it is not None:
            if it not in its:
                raise IterationError(f"Iteration {it} for Variable {var} not in {self}")
            return it
        times: "NDArray[float]" = self.get_time(its)
        return its[times.searchsorted(time)]

    def get_coords(self, region: str, it: int, rls: "NDArray[int]", exclude_ghosts: int = 0) -> dict[int, "NDArray[float]"]:

        if len(region) > 1:
            return {rl: [ori + dx * np.arange(exclude_ghosts, nn-exclude_ghosts)
                         for ori, dx, nn in zip(*self._structure[it][rl][region])]
                    for rl in rls}

        ret = {}
        for rl in rls:
            ori, dx, nn = self._structure[it][rl][region]
            ret[rl] = [ori + dx * np.arange(exclude_ghosts, nn-exclude_ghosts)]
        return ret
