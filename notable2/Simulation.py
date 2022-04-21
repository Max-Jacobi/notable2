import os
import re
from os.path import basename, isfile
from typing import Optional, Type, Callable, overload, Any, TYPE_CHECKING, Dict
from collections.abc import Iterable
import numpy as np
from scipy.signal import find_peaks  # type: ignore
from scipy.interpolate import interp2d  # type: ignore
from scipy.optimize import minimize  # type: ignore
from h5py import File as HDF5  # type: ignore

from .DataHandlers import DataHandler
from .EOS import EOS, TabulatedEOS
from .RCParams import rcParams
from .Variable import Variable, GridFuncVariable, TimeSeriesVariable
from .Variable import PPGridFuncVariable, PPTimeSeriesVariable, GravitationalWaveVariable
from .PostProcVariables import get_pp_variables
from .Plot import plotGD, plotTS, animateGD, plotHist
from .Animations import GDAniFunc as GDAF
from .Animations import TSLineAniFunc as TSAF
from .Utils import IterationError, VariableError, BackupException, RLArgument

if TYPE_CHECKING:
    from numpy.typing import NDArray


class Simulation():
    """Documentation for Simulation """
    sim_path: str
    sim_name: str
    nice_name: str
    t_merg: Optional[float]
    ADM_M: Optional[float]
    ADM_J: Optional[float]
    rls: Dict[int, 'NDArray[np.int_]']
    finest_rl: Dict[int, int]
    data_handler: DataHandler
    eos: EOS
    is_cartoon: bool
    verbose: int = 0
    pp_hdf5_path: str
    pp_grid_func_variables: Dict[str, Dict[str, Any]]
    pp_time_series_variables: Dict[str, Dict[str, Any]]
    pp_gw_variables: Dict[str, Dict[str, Any]]
    plotGD: Callable
    plotTS: Callable
    animateGD: Callable
    plotHist: Callable

    def __init__(self,
                 sim_path: str,
                 data_handler: Optional[Type[DataHandler]] = None,
                 eos_path: Optional[str] = None,
                 offset: Optional[Dict[str, float]] = None,
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
        self.rls = {it: np.arange(max(struc.keys())+1) for it, struc in self._structure.items()}
        self.finest_rl = {it: rls.max() for it, rls in self.rls.items()}

        self.pp_grid_func_variables = {}
        for ufile in rcParams.PPGridFuncVariable_files:
            self.pp_grid_func_variables.update(get_pp_variables(ufile, self.eos))
        self.pp_time_series_variables = {}
        for ufile in rcParams.PPTimeSeries_files:
            self.pp_time_series_variables.update(get_pp_variables(ufile, self.eos))
        self.pp_gw_variables = {}
        for ufile in rcParams.PPGW_files:
            self.pp_gw_variables.update(get_pp_variables(ufile, self.eos))

        self.pp_hdf5_path = f"{self.sim_path}/PPVars"
        if not os.path.isdir(self.pp_hdf5_path):
            os.mkdir(self.pp_hdf5_path)

        self.ADM_M, self.ADM_J = self.get_ADM_MJ() if not self.is_cartoon else (None, None)
        self.t_merg = self.get_t_merg() if not self.is_cartoon else None

    def __repr__(self):
        return f"Einstein Toolkit simulation {self.sim_name}"

    def __str__(self):
        return self.sim_name

    def expand_rl(self, rls: RLArgument, it: int) -> 'NDArray[np.int_]':
        """Expand the "rl" argument to a valid array of refinementlevels"""
        sim_rls = np.array(list(self._structure[it].keys()))
        if rls is None:
            return sim_rls
        if not isinstance(rls, Iterable):
            rls = np.array([rls, ])
        if ... in rls:
            rls = list(rls)
            el_ind = rls.index(...)
            rls.remove(...)
            rls = np.array(rls)
            rls[rls < 0] += sim_rls.max() + 1
            if el_ind == 0:
                bounds = sorted([0, rls[0]])
            elif el_ind == len(rls):
                bounds = sorted([rls[-1], sim_rls.max()])
            else:
                bounds = sorted([rls[el_ind-1], rls[el_ind]])

            el_ex = list(range(bounds[0], bounds[1]+1))
            rls = list(rls)
            rls = np.sort(rls[:el_ind-1] + el_ex + rls[el_ind+1:])
        rls = np.array(rls)
        rls[rls < 0] += sim_rls.max() + 1
        return np.sort(rls)

    @overload
    def get_time(self, it: int) -> float:
        ...

    @overload
    def get_time(self, it: 'NDArray[np.int_]') -> 'NDArray[np.float_]':
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
        return self._times[self._its == it][0]

    @overload
    def get_restart(self, it: int) -> int:
        ...

    @overload
    def get_restart(self, it: 'NDArray[np.int_]') -> 'NDArray[np.int_]':
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
    def get_it(self, time: 'NDArray[np.float_]') -> 'NDArray[np.int_]':
        ...

    def get_it(self, time):
        """Get the smallest itteration(s) with time < the given time"""
        return self._its[self._times.searchsorted(time)]

    def get_t_merg(self) -> Optional[float]:
        try:
            habs = self.get_data("h-abs")
            ind = find_peaks(habs.data, height=.15)[0][0]
            # ind = np.argmax(habs.data)
            return habs.times[ind]
        except VariableError:
            if self.verbose:
                print(f"{self.name} using minumum of alpha to determine merger time")
            data = self.get_data('alpha-min')
            times = data.times
            peaks = find_peaks(-data.data)[0]
            diffs = np.diff(data.data[peaks])
            if np.any(big_diffs := diffs < -.05):
                min_ind = peaks[np.argwhere(big_diffs)[0][0]+1]
                return float(times[min_ind])

    def get_ADM_MJ(self):
        pattern = r"ADM mass of the system : ([0-9]\.[0-9]+) M_sol\n"
        pattern += r" *Total angular momentum : ([0-9]\.[0-9]+) G M_sol\^2 / c"

        if not isfile(outf := f"{self.sim_path}/output-0000/{self.sim_name}.out"):
            return None, None
        with open(outf, 'r') as file:
            m = re.search(pattern, file.read())
        return float(m[1]), float(m[2])

    def get_offset(self, it: int) -> 'NDArray[np.float_]':
        if self.is_cartoon:
            return np.array([0., 0.])
        frl = self.finest_rl[it]
        coords = self.get_coords('xy', it)[frl]
        alp_dat = self.get_data('alpha', region='xy', it=it)[frl]

        return minimize(
            lambda xy: interp2d(coords['x'], coords['y'], alp_dat.T, kind='quintic')(*xy)[0],
            np.array([0., 0.]))['x']

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
            if key in self.pp_gw_variables:
                return GravitationalWaveVariable(key=key, sim=self, **self.pp_gw_variables[key])
            raise VariableError(f"Could not find key {key} in {self}.") from last_exc

    def get_data(self, key: str, **kwargs):
        return self.get_variable(key).get_data(**kwargs)

    def get_coords(self,
                   region: str,
                   it: int,
                   exclude_ghosts: int = 0
                   ) -> Dict[int, Dict[str, 'NDArray[np.float_]']]:

        if len(region) > 1:
            ret: Dict[int, Dict[str, 'NDArray[np.float_]']] = {rl: {} for rl in self._structure[it].keys()}
            for rl in ret.keys():
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
        for rl in self._structure[it].keys():
            ori, dx, nn = self._structure[it][rl][region]
            ret[rl] = {region: ori + dx * np.arange(exclude_ghosts, nn-exclude_ghosts)}
        return ret

    # def delete_saved_pp_variable(self, key: str):
    #     with HDF5(self.pp_hdf5_path, 'a') as hf:
    #         to_delete = [kk for kk in hf if key in kk]
    #         for kk in to_delete:
    #             del hf[kk]
    #             hf.flush()

    # def rename_saved_pp_variable(self, key: str,  new_key: str):
    #     with HDF5(self.pp_hdf5_path, 'a') as hf:
    #         to_rename = {kk: kk.replace(key, new_key) for kk in hf if key in kk}
    #         for kk, nk in to_rename.items():
    #             hf[nk] = hf[kk][:]
    #             del hf[kk]
    #         hf.flush()

    def GDAniFunc(self, *args, **kwargs):
        return GDAF(self, *args, **kwargs)

    def TSAniFunc(self, *args, **kwargs):
        return TSAF(self, *args, **kwargs)


Simulation.plotGD = plotGD
Simulation.plotTS = plotTS
Simulation.animateGD = animateGD
Simulation.plotHist = plotHist
