from os.path import basename
from typing import Optional, Type, Callable, overload, Any
from collections.abc import Iterable
import numpy as np


from .DataHandlers import DataHandler
from .EOS import EOS, TabulateEOS
from .RCParams import rcParams
from .Variable import Variable, GridDataVariable, TimeSeriesVariable, UGridDataVariable, UTimeSeriesVariable, UserVariable
from .UserVariables import get_user_variables
from .Plot import plotGD, plotTS
from .Utils import IterationError, VariableError, BackupException, RLArgument, NDArray


class Simulation():
    """Documentation for Simulation """
    sim_path: str
    sim_name: str
    rls: NDArray[np.int_]
    data_handler: DataHandler
    eos: EOS
    plotGD: Callable
    plotTS: Callable
    is_cartoon: bool
    verbose: bool = False
    ud_hdf5_path: str
    user_grid_data_variables: dict[str, dict[str, Any]]
    user_time_series_variables: dict[str, dict[str, Any]]

    def __init__(self,
                 sim_path: str,
                 data_handler: Optional[Type[DataHandler]] = None,
                 eos_path: Optional[str] = None,
                 offset: Optional[dict[str, float]] = None,
                 is_cartoon: bool = False
                 ):
        self.sim_path = sim_path
        self.sim_name = basename(sim_path)
        self.is_cartoon = is_cartoon

        self.data_handler = data_handler(self) if data_handler is not None \
            else rcParams.default_data_handler(self)
        if eos_path == 'ideal':
            ...
        else:
            self.eos = TabulateEOS(eos_path if eos_path is not None else rcParams.default_eos_path)
        self._offset = offset if offset is not None else dict(x=0, y=0, z=0)

        (self._its, self._times, self._restarts), self._structure, self.its_lookup \
            = self.data_handler.get_structure()
        self.rls = np.array(list(self._structure[0].keys()))
        self.finest_rl = self.rls.max()

        self.user_grid_data_variables = {}
        for ufile in rcParams.UGridDataVariable_files:
            self.user_grid_data_variables.update(get_user_variables(ufile))
        self.user_time_series_variables = {}
        for ufile in rcParams.UTimeSeries_files:
            self.user_time_series_variables.update(get_user_variables(ufile))

        self.ud_hdf5_path = f"{self.sim_path}/{self.sim_name}_UD.hdf5"

    def __repr__(self):
        return f"Einstein Toolkit simulation {self.sim_name}"

    def __str__(self):
        return self.sim_name

    def expand_rl(self, rls: RLArgument) -> NDArray[np.int_]:
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

    @overload
    def get_time(self, it: int) -> float:
        ...

    @overload
    def get_time(self, it: NDArray[np.int_]) -> NDArray[np.float_]:
        ...

    def get_time(self, it):
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

    @overload
    def get_restart(self, it: int) -> int:
        ...

    @overload
    def get_restart(self, it: NDArray[np.int_]) -> NDArray[np.int_]:
        ...

    def get_restart(self, it):
        """Get the restart number for iteration(s) it"""
        if isinstance(it, np.ndarray):
            if any(ii not in self._its for ii in it):
                it_er = self._its[np.array([ii not in self._its for ii in it])]
                raise IterationError(f"Iterations {it_er} not all in {self.sim_name}")
            return self._times[self._its.searchsorted(it)]
        if it not in self._its:
            raise IterationError(f"Iteration {it} not all in {self.sim_name}")
        return self._restarts[np.argwhere(self._its)]

    @overload
    def get_it(self, time: float) -> int:
        ...

    @overload
    def get_it(self, time: NDArray[np.float_]) -> NDArray[np.int_]:
        ...

    def get_it(self, time):
        """Get the smallest itteration(s) with time < the given time"""
        return self._its[self._times.searchsorted(time)]

    def get_variable(self, key: str) -> Variable:
        """Return Variable for key"""
        for var in [TimeSeriesVariable, GridDataVariable]:
            try:
                return var(key, self)
            except BackupException as bexc:
                for bu_var in bexc.backups:
                    try:
                        if self.verbose:
                            print(f"Trying backup {bu_var.key} instead of {key}")
                        return bu_var
                    except VariableError as exc:
                        last_exc = exc
                        continue
            except VariableError as exc:
                last_exc = exc
                continue
        else:
            if key in self.user_grid_data_variables:
                return UGridDataVariable(key, self, **self.user_grid_data_variables[key])
            if key in self.user_time_series_variables:
                return UTimeSeriesVariable(key, self, **self.user_time_series_variables[key])
            raise VariableError(f"Could not find key {key} in {self}.") from last_exc

    def get_coords(self,
                   region: str,
                   it: int,
                   exclude_ghosts: int = 0
                   ) -> dict[int, dict[str, NDArray[np.float_]]]:

        if len(region) > 1:
            return {rl: {ax: of + ori + dx * np.arange(exclude_ghosts, nn-exclude_ghosts)
                         for ax, (ori, dx, nn), of in zip(region,
                                                          zip(*self._structure[it][rl][region]),
                                                          [self._offset[ax] for ax in region]
                                                          )}
                    for rl in self.rls}

        ret = {}
        for rl in self.rls:
            ori, dx, nn = self._structure[it][rl][region]
            ret[rl] = {region: ori + dx * np.arange(exclude_ghosts, nn-exclude_ghosts)}
        return ret


Simulation.plotGD = plotGD
Simulation.plotTS = plotTS
