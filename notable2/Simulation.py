import os
import re
from os.path import basename, isfile
from typing import Optional, Type, Callable, overload, Any, TYPE_CHECKING, Dict
from collections.abc import Iterable
from time import sleep
from types import MethodType
import json

import numpy as np
from scipy.signal import find_peaks  # type: ignore
from scipy.interpolate import interp1d, interp2d  # type: ignore
from scipy.optimize import minimize, bisect  # type: ignore
from h5py import File as HDF5  # type: ignore

from .DataHandlers import DataHandler
from .Config import config
from .Variable import GridFuncBaseVariable, TimeSeriesBaseVariable, Variable, GridFuncVariable, TimeSeriesVariable
from .Variable import PPGridFuncVariable, PPTimeSeriesVariable, GravitationalWaveVariable
from .PostProcVariables import get_pp_variables
from .Utils import IterationError, VariableError, BackupException, RLArgument
from .Plot import plotGD, plotTS, animateGD, plotHist
from .Animations import GDAniFunc, TSLineAniFunc


from tabulatedEOS.PizzaEOS import PizzaEOS

if TYPE_CHECKING:
    from numpy.typing import NDArray


def _get_sim_path(sim_path):
    cactus_base = (os.environ['CACTUS_BASEDIR']
                   if "CACTUS_BASEDIR" in os.environ else None)
    if cactus_base is None or sim_path[0] == '/':
        return sim_path
    if sim_path in os.listdir(cactus_base):
        return os.path.join(cactus_base, sim_path)
    if os.path.isdir(f"{cactus_base}/by-short-name") and sim_path in os.listdir(f"{cactus_base}/by-short-name"):
        return f"{cactus_base}/by-short-name/{sim_path}"
    raise ValueError(
        f"Could not find {sim_path} in {cactus_base} or {cactus_base}/by-short-name")


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
    eos: PizzaEOS
    is_cartoon: bool
    verbose: int = 0
    pp_hdf5_path: str
    pp_grid_func_variables: Dict[str, Dict[str, Any]]
    pp_time_series_variables: Dict[str, Dict[str, Any]]
    pp_gw_variables: Dict[str, Dict[str, Any]]
    properties: Dict[str, Any]
    plotGD: Callable
    plotTS: Callable
    plotHist: Callable
    animateGD: Callable
    plotHist: Callable

    def __init__(self,
                 sim_path: str,
                 data_handler: Optional[Type[DataHandler]] = None,
                 eos_path: Optional[str] = None,
                 offset: Optional[Dict[str, float]] = None,
                 is_cartoon: bool = False
                 ):
        self.sim_path = _get_sim_path(sim_path)
        self.sim_name = basename(sim_path)
        self.nice_name = self.sim_name
        self.is_cartoon = is_cartoon
        self.properties = dict()

        self.data_handler = data_handler(self) if data_handler is not None \
            else config.default_data_handler(self)
        # TODO: Add support for other EOSs by creating a EOS factory based on
        # the content in the path
        self.eos = PizzaEOS(eos_path)

        self._offset = offset if offset is not None else dict(x=0, y=0, z=0)

        (self._its, self._times, self._restarts), self._structure, self.its_lookup \
            = self.data_handler.get_structure()
        self.rls = {it: np.arange(max(struc.keys())+1)
                    for it, struc in self._structure.items()}
        self.finest_rl = {it: rls.max() for it, rls in self.rls.items()}

        self.pp_grid_func_variables = {}
        self.pp_time_series_variables = {}
        self.pp_gw_variables = {}
        self.read_PPVariables()
        self.read_properties()

        self.pp_hdf5_path = f"{self.sim_path}/PPVars"
        if not os.path.isdir(self.pp_hdf5_path):
            os.mkdir(self.pp_hdf5_path)

        self.ADM_M, self.ADM_J = self.get_ADM_MJ() if not self.is_cartoon else (None, None)
        try:
            if self.is_cartoon:
                self.t_merg = None
            else:
                self.t_merg = self.get_t_merg()
        except (ValueError, KeyError, OSError):
            self.t_merg = None

        self.plotGD = MethodType(plotGD, self)
        self.plotTS = MethodType(plotTS, self)
        self.plotHist = MethodType(plotHist, self)
        self.animateGD = MethodType(animateGD, self)
        self.GDAniFunc = MethodType(GDAniFunc, self)
        self.TSLineAniFunc = MethodType(TSLineAniFunc, self)

    def __repr__(self):
        return f"Einstein Toolkit simulation {self.sim_name}"

    def __str__(self):
        return self.sim_name

    def read_PPVariables(self):
        for ufile in config.PPGridFuncVariable_files:
            self.pp_grid_func_variables.update(
                get_pp_variables(ufile, self.eos))
        for ufile in config.PPTimeSeries_files:
            self.pp_time_series_variables.update(
                get_pp_variables(ufile, self.eos))
        for ufile in config.PPGW_files:
            self.pp_gw_variables.update(get_pp_variables(ufile, self.eos))

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
                it_er = np.array([ii for ii in it if ii not in self._its])
                raise IterationError(
                    f"Iterations {it_er} not all in {self.sim_name}")
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
                raise IterationError(
                    f"Iterations {it_er} not all in {self.sim_name}")
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
        "Get the smallest itteration(s) with time < the given time"
        return self._its[self._times.searchsorted(time, side='left')]

    def get_t_merg(self, use_GW: bool = True) -> Optional[float]:
        """
        Get the merger time
        if use_GW is True (default True) use the largest peak in abs(h) (minumum height 0.1)
        if use_GW is False or there is no peak in GW data use first large decrease of the
        minium of the lapse function (> 0.05)
        returns None if no large decrease in min(lapse) (no merger happened)
        """
        if use_GW:
            for n_try in range(10):
                try:
                    habs = self.get_data("h-abs")
                    ind, prop = find_peaks(habs.data, height=.1)
                    if len(ind) > 0:
                        return habs.times[ind[np.argmax(prop['peak_heights'])]]
                    else:
                        break
                except OSError as ex:
                    sleep(5)
                    continue
                except (VariableError, IndexError):
                    break
        if self.verbose:
            print(
                f"{self.sim_name} using peaks in the "
                "lapse minimum to determine merger time"
            )
        dat = self.get_data('alpha-min')
        i_peaks = find_peaks(dat.data)[0]
        i_dips = find_peaks(-dat.data)[0]
        diffs = dat.data[i_peaks] - dat.data[i_dips]
        if np.any(big_diffs := diffs > 0.05):
            return dat.times[i_dips][big_diffs][0]
        else:
            if self.verbose:
                print(
                    f"{self.sim_name}: no dips in the lapse "
                    "minimum to determine merger time"
                )
            return

    def get_ADM_MJ(self):
        pattern = r"ADM mass of the system : ([0-9]\.[0-9]+) M_sol\n"
        pattern += r" *Total angular momentum : ([0-9]\.[0-9]+) G M_sol\^2 / c"

        if not isfile(outf := f"{self.sim_path}/output-0000/{self.sim_name}.out"):
            return None, None
        for n_try in range(20):
            try:
                with open(outf, 'r') as file:
                    m = re.search(pattern, file.read())
                break
            except KeyboardInterrupt:
                raise
            except Exception as ex:
                sleep(5)
                continue
        else:
            print("could not get ADM mass")
            raise ex

        if m is not None:
            return float(m[1]), float(m[2])
        return np.nan, np.nan

    def get_offset(self, it: int) -> 'NDArray[np.float_]':
        if self.is_cartoon:
            return np.array([0., 0.])
        frl = self.finest_rl[it]
        coords = self.get_coords('xy', it)[frl]
        alp_dat = self.get_data('alpha', region='xy', it=it)[frl]

        return minimize(
            lambda xy: interp2d(
                coords['x'], coords['y'], alp_dat.T, kind='quintic')(*xy)[0],
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
            for var, var_list in (
                (PPGridFuncVariable, self.pp_grid_func_variables),
                (PPTimeSeriesVariable, self.pp_time_series_variables),
                (GravitationalWaveVariable, self.pp_gw_variables)
            ):
                if key in var_list:
                    return var(key=key, sim=self, **var_list[key])

            raise VariableError(
                f"Could not find key {key} in {self}.") from last_exc

    def get_data(self, key: str,
                 time: Optional[float] = None,
                 min_time: Optional[float] = None,
                 max_time: Optional[float] = None,
                 **kwargs):
        var = self.get_variable(key)
        if time is not None:
            its = var.get_it(time=time, **kwargs)
            return var.get_data(it=its, **kwargs)
        elif min_time is not None or max_time is not None:
            if not isinstance(var, TimeSeriesBaseVariable):
                raise ValueError(
                    f"{var} is not a time series. Can not use min_time or max_time.")
            its = var.available_its(**kwargs)
            if min_time is not None:
                min_it = var.get_it(min_time, **kwargs)
                its = its[its >= min_it]
            if max_time is not None:
                max_it = var.get_it(max_time, **kwargs)
                its = its[its <= max_it]
            return var.get_data(it=its, **kwargs)
        return var.get_data(**kwargs)

    def get_coords(self,
                   region: str,
                   it: int,
                   exclude_ghosts: int = 0
                   ) -> Dict[int, Dict[str, 'NDArray[np.float_]']]:

        if len(region) > 1:
            ret: Dict[int, Dict[str, 'NDArray[np.float_]']] = {
                rl: {} for rl in self._structure[it].keys()}
            for rl in ret.keys():
                if region not in self._structure[it][rl]:
                    for ax in region:
                        ret[rl][ax] = np.array([])
                else:
                    for ax, ori, dx, nn, of in zip(
                        region,
                        *self._structure[it][rl][region],
                        [self._offset[ax] for ax in region]
                    ):
                        ret[rl][ax] = of + ori + dx * \
                            np.arange(exclude_ghosts, nn-exclude_ghosts)
            return ret

        ret = {}
        for rl in self._structure[it].keys():
            ori, dx, nn = self._structure[it][rl][region]
            ret[rl] = {region: ori + dx *
                       np.arange(exclude_ghosts, nn-exclude_ghosts)}
        return ret

    def delete_saved_pp_time_series(self, key: str):
        with HDF5(f"{self.pp_hdf5_path}/time_series.h5", 'r+') as hf:
            to_delete = [kk for kk in hf if key in kk]
            for kk in to_delete:
                print(f"Deleting {kk} in {self}")
                del hf[kk]
                hf.flush()

    def rename_saved_pp_time_series(self, key: str,  new_key: str):
        with HDF5(f"{self.pp_hdf5_path}/time_series.h5", 'r+') as hf:
            to_rename = {kk: kk.replace(key, new_key)
                         for kk in hf if key in kk}
            for kk, nk in to_rename.items():
                hf[nk] = hf[kk][()]
                hf[nk].attrs.update(hf[kk].attrs)
                del hf[kk]
            hf.flush()

    def write_properties(self, **properties):
        """
        Write properties json file containing the given properties.
        """
        with open(f"{self.sim_path}/properties.json", 'w') as file:
            json.dump(properties, file, indent=4)
        self.read_properties()

    def read_properties(self):
        """
        Read properties json file
        """
        if not os.path.isfile(f"{self.sim_path}/properties.json"):
            return

        with open(f"{self.sim_path}/properties.json", 'r') as file:
            self.properties = json.load(file)

        # if property is a attribute of simulation and has a fitting type set it
        for key in list(self.properties):
            if hasattr(self, key) and isinstance(self.properties[key], type(getattr(self, key))):
                setattr(self, key, self.properties.pop(key))
