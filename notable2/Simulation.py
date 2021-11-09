import os
from os.path import basename, isdir
from typing import Optional, Type, Callable, overload, Any, Union
from collections.abc import Iterable
import numpy as np
from numpy.typing import NDArray
from scipy.signal import find_peaks  # type: ignore
from h5py import File as HDF5  # type: ignore

from .DataHandlers import DataHandler
from .EOS import EOS, TabulatedEOS
from .RCParams import rcParams
from .Variable import Variable, GridFuncVariable, TimeSeriesVariable, PPGridFuncVariable, PPTimeSeriesVariable
from .DataObjects import GridFunc, TimeSeries
from .PostProcVariables import get_pp_variables
from .Plot import plotGD, plotTS, animateGD, plotHist
from .Animation import GDAniFunc as GDAF
from .Utils import IterationError, VariableError, BackupException, RLArgument


class Simulation():
    """Documentation for Simulation """
    sim_path: str
    sim_name: str
    nice_name: str
    t_merg: Optional[float]
    rls: NDArray[np.int_]
    data_handler: DataHandler
    eos: EOS
    is_cartoon: bool
    verbose: int = 0
    pp_hdf5_path: str
    pp_grid_func_variables: dict[str, dict[str, Any]]
    pp_time_series_variables: dict[str, dict[str, Any]]
    plotGD: Callable
    plotTS: Callable
    animateGD: Callable
    plotHist: Callable

    def __init__(self,
                 sim_path: str,
                 data_handler: Optional[Type[DataHandler]] = None,
                 eos_path: Optional[str] = None,
                 offset: Optional[dict[str, float]] = None,
                 is_cartoon: bool = False
                 ):
        cactus_base = (os.environ['CACTUS_BASEDIR'] if "CACTUS_BASEDIR" in os.environ else None)
        self.sim_path = (f"{cactus_base}/{sim_path}"
                         if sim_path[0] != '/' and cactus_base is not None else sim_path)
        self.sim_name = basename(sim_path)
        self.nice_name = self.sim_name
        self.is_cartoon = is_cartoon

        self.data_handler = data_handler(self) if data_handler is not None \
            else rcParams.default_data_handler(self)
        if eos_path is None:
            self.eos = TabulatedEOS(rcParams.default_eos_path)
        elif eos_path == 'ideal':
            ...
        else:
            if eos_path[0] != '/' and cactus_base is not None:
                eos_path = f"{cactus_base}/EOSs/{eos_path}"
            self.eos = TabulatedEOS(eos_path)

        self._offset = offset if offset is not None else dict(x=0, y=0, z=0)

        (self._its, self._times, self._restarts), self._structure, self.its_lookup \
            = self.data_handler.get_structure()
        self.rls = np.array(list(self._structure[0].keys()))
        self.finest_rl = self.rls.max()

        self.pp_grid_func_variables = {}
        for ufile in rcParams.PPGridFuncVariable_files:
            self.pp_grid_func_variables.update(get_pp_variables(ufile, self.eos))
        self.pp_time_series_variables = {}
        for ufile in rcParams.PPTimeSeries_files:
            self.pp_time_series_variables.update(get_pp_variables(ufile, self.eos))

        self.pp_hdf5_path = f"{self.sim_path}/{self.sim_name}_PP.hdf5"

        self.t_merg = self.get_t_merg() if not self.is_cartoon else None

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
        if ... in rls:
            rls = list(rls)
            el_ind = rls.index(...)
            rls.remove(...)
            rls = np.array(rls)
            rls[rls < 0] += self.finest_rl + 1
            if el_ind == 0:
                bounds = sorted([0, rls[0]])
            elif el_ind == len(rls):
                bounds = sorted([rls[-1], self.rls.max()])
            else:
                bounds = sorted([rls[el_ind-1], rls[el_ind]])

            el_ex = list(range(bounds[0], bounds[1]+1))
            rls = list(rls)
            rls = np.sort(rls[:el_ind-1] + el_ex + rls[el_ind+1:])
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

    def get_t_merg(self) -> float:
        data = self.get_data('alpha_min')
        times = data.times
        peaks = find_peaks(-data.data)[0]
        min_ind = peaks[np.argmax(np.abs(np.diff(data.data[peaks])))+1]
        return float(times[min_ind])

    def get_variable(self, key: str) -> Variable:
        """Return Variable for key"""
        for var in [TimeSeriesVariable, GridFuncVariable]:
            try:
                return var(key, self)
            # except BackupException as bexc:
                # for bu_var in bexc.backups:
                # return bu_var
            except (BackupException, VariableError) as exc:
                last_exc = exc
                continue
        else:
            if key in self.pp_grid_func_variables:
                return PPGridFuncVariable(key=key, sim=self, **self.pp_grid_func_variables[key])
            if key in self.pp_time_series_variables:
                return PPTimeSeriesVariable(key=key, sim=self, **self.pp_time_series_variables[key])
            raise VariableError(f"Could not find key {key} in {self}.") from last_exc

    def get_data(self, key: str, **kwargs):
        return self.get_variable(key).get_data(**kwargs)

    def get_coords(self,
                   region: str,
                   it: int,
                   exclude_ghosts: int = 0
                   ) -> dict[int, dict[str, NDArray[np.float_]]]:

        if len(region) > 1:
            ret: dict[int, dict[str, NDArray[np.float_]]] = {rl: {} for rl in self.rls}
            for rl in self.rls:
                if region not in self._structure[it][rl]:
                    for ax in region:
                        ret[rl][ax] = np.array([])
                else:
                    for ax, ori, dx, nn, of in zip(region,
                                                   *self._structure[it][rl][region],
                                                   [self._offset[ax] for ax in region]):
                        ret[rl][ax] = of + ori + dx * np.arange(exclude_ghosts, nn-exclude_ghosts)
            return ret

        ret = {}
        for rl in self.rls:
            ori, dx, nn = self._structure[it][rl][region]
            ret[rl] = {region: ori + dx * np.arange(exclude_ghosts, nn-exclude_ghosts)}
        return ret

    def delete_saved_pp_variable(self, key: str):
        with HDF5(self.pp_hdf5_path, 'a') as hf:
            to_delete = [kk for kk in hf if key in kk]
            for kk in to_delete:
                del hf[kk]
                hf.flush()

    def GDAniFunc(self, *args, **kwargs):
        return GDAF(self, *args, **kwargs)


Simulation.plotGD = plotGD
Simulation.plotTS = plotTS
Simulation.animateGD = animateGD
Simulation.plotHist = plotHist
