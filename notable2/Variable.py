"""
Variable objects Inheritance tree:

- Variable
- UVariable

- GridFuncVariable(Variable)
- PPGridFuncVariable(Variable)
- TimeSeriesVariable(Variable)
- PPTimeSeries(Variable)
- ?Reduction seperate or as part of PPTimeSeries?
"""

from abc import ABC, abstractmethod
from typing import (TYPE_CHECKING, Callable, List, Dict,
                    Optional, Union, Any, overload)
from functools import reduce
import re
import json
import numpy as np

from .Config import config
from .DataObjects import GridFunc, TimeSeries, PPGridFunc, PPTimeSeries, GWData
from .Utils import PlotName, Units, VariableError, BackupException, IterationError

if TYPE_CHECKING:
    from .Utils import Simulation
    from numpy.typing import NDArray


class Variable(ABC):
    """Documentation for Variable"""
    sim: "Simulation"
    key: str
    vtype: str
    plot_name: PlotName
    kwargs: Dict[str, Any]
    scale_factor: float
    backups: List["Variable"]
    get_data: Callable
    available_its: Callable
    get_it: Callable

    def __init__(self, key: str, sim: "Simulation", ):
        self.sim = sim
        self.key = key

    def __repr__(self):
        return f"{self.__class__.__name__} {self.key}"

    def __str__(self):
        return f"{self.key}"


class TimeSeriesBaseVariable(Variable, ABC):
    """ABC for TimeSeriesVariables"""
    vtype: str = 'time'

    @overload
    def get_data(self, it: int, **kwargs) -> float:
        ...

    @overload
    def get_data(self, it: Optional['NDArray[np.int_]'], **kwargs) -> TimeSeries:
        ...

    @abstractmethod
    def get_data(self, it: Optional[Union[int, 'NDArray[np.int_]']], **kwargs):
        """Returns TimeSeries DataObject.
        If it is an int the value at that iteration is returned instead.
        If it is an array only those iterations are returned"""
        ...

    @abstractmethod
    def available_its(self, **kwargs) -> 'NDArray[np.int_]':
        """Returns Array of available iterations"""
        ...

    def get_it(self, time: float, t_merg: bool = True, **kwargs):
        its = self.available_its(**kwargs)

        times = self.sim.get_time(its)
        if self.sim.t_merg is not None and t_merg:
            times -= self.sim.t_merg
        if time > (max_time := times.max()):
            time = max_time
        if time < (min_time := times.min()):
            time = min_time
        it = its[times.searchsorted(time)]

        return it


class GridFuncBaseVariable(Variable, ABC):
    """ABC for TimeSeriesVariables"""
    vtype: str = 'grid'

    @abstractmethod
    def get_data(self,
                 region: str,
                 it: int,
                 exclude_ghosts: int,
                 **kwargs) -> GridFunc:
        """Returns GridFunc DataObject for given it and region.
        exclude_ghosts number of cells are cut of."""
        ...

    @abstractmethod
    def available_its(self, region: str, **kwargs) -> 'NDArray[np.int_]':
        """Returns Array of available iterations"""
        ...

    def get_it(self, time: float, region: str, t_merg: bool = True, **kwargs):
        its = self.available_its(region, **kwargs)

        times = self.sim.get_time(its)
        if self.sim.t_merg is not None and t_merg:
            times -= self.sim.t_merg
        if time > (max_time := times.max()):
            time = max_time
        if time < (min_time := times.min()):
            time = min_time
        it = its[times.searchsorted(time)]

        return it


class NativeVariable(Variable):
    """ABC for Native variables"""
    file_key: str
    file_name: str
    alias: List[str]

    def __init__(self, key: str, sim: "Simulation", json_path: str):
        super().__init__(key, sim)

        with open(json_path, 'r') as ff:
            json_dic = json.load(ff)
            try:
                key_dict = json_dic[key]
            except KeyError as exc:

                for kk, dic in json_dic.items():
                    if (match := re.fullmatch(kk, key)) is not None:
                        key_dict = dic
                        for ii, val in enumerate(match.groups()):
                            if 'format_opt' not in key_dict:
                                key_dict['format_opt'] = {}
                            key_dict['format_opt'][f'key{ii}'] = val
                        if self.sim.verbose > 1:
                            print(
                                f"{self.sim.sim_name}: Using regex key {kk} for {key}")
                        break
                    if 'alias' in dic and key in dic['alias']:
                        if self.sim.verbose > 1:
                            print(
                                f"{self.sim.sim_name}: Using aliased key {kk} for {key}")
                        key_dict = dic
                        break
                else:
                    raise VariableError(
                        f"Key {key} not found in {json_path}") from exc

        self.file_key = key_dict.pop('file_key')
        self.file_name = key_dict.pop('file_name')
        self.scale_factor = key_dict.pop(
            'scale_factor') if 'scale_factor' in key_dict else 1
        if isinstance(self.scale_factor, str):
            self.scale_factor = Units[self.scale_factor]
        self.kwargs = key_dict.pop('kwargs') if 'kwargs' in key_dict else {}
        self.alias = key_dict.pop('alias') if 'alias' in key_dict else []
        self.backups = []
        if 'backups' in key_dict:
            for bu in key_dict.pop('backups'):
                try:
                    self.backups.append(self.sim.get_variable(bu))
                except VariableError:
                    continue
        self.plot_name = PlotName(**key_dict)


class PostProcVariable(Variable):
    """ABC for post processed Variables"""

    dependencies: List[Variable]
    PPkeys: Union[List[str], Dict[str, Any]]

    def __init__(self,
                 key: str,
                 sim: "Simulation",
                 dependencies: List[str],
                 func: Callable,
                 plot_name_kwargs: Dict[str, Any],
                 backups: Optional[List[str]] = None,
                 scale_factor: float = 1,
                 save: bool = True,
                 kwargs: Optional[Dict[str, Any]] = None,
                 PPkeys: Optional[Union[List[str], Dict[str, Any]]] = None,
                 ):

        super().__init__(key, sim)
        self.backups = []
        if backups is not None:
            for bu in backups:
                try:
                    self.backups.append(self.sim.get_variable(bu))
                except VariableError:
                    continue

        self.kwargs = kwargs if kwargs is not None else {}
        self.PPkeys = PPkeys if PPkeys is not None else {}
        self.func = func
        self.scale_factor = scale_factor
        self.save = save
        if isinstance(self.scale_factor, str):
            self.scale_factor = Units[self.scale_factor]
        self.plot_name = PlotName(**plot_name_kwargs)

        try:
            self.dependencies = [sim.get_variable(key) for key in dependencies]
        except VariableError as exc:
            if any(self.backups):
                raise BackupException(self.backups) from exc
            raise exc

    def _available_its(self, region: str, **kwargs) -> 'NDArray[np.int_]':
        try:
            its = []
            for dep in self.dependencies:
                if dep.vtype == 'grid':
                    its.append(dep.available_its(region, **kwargs))
                else:
                    its.append(dep.available_its(**kwargs))
            its = reduce(np.intersect1d, its)
            if len(its) == 0:
                raise IterationError(f"No common iterations found for {self}")
            return np.array(its, dtype=int)
        except VariableError:
            for bvar in self.backups:
                try:
                    if bvar.vtype == 'grid':
                        its = bvar.available_its(region, **kwargs)
                    else:
                        its = bvar.available_its(**kwargs)
                    if self.sim.verbose > 1:
                        print(
                            f"{self.sim.sim_name}: Using {bvar.key} instead of {dep.key}")
                    return its
                except (VariableError, IterationError):
                    continue
            raise


class GridFuncVariable(NativeVariable, GridFuncBaseVariable):
    """Variable for native grid functions"""

    def __init__(self, key: str, sim: "Simulation", ):
        super().__init__(key, sim, config.GridFuncVariable_json)

    def get_data(
        self,
        region: str,
        it: int,
        exclude_ghosts: int = 0,
        **kwargs
        ) -> GridFunc:

        coords = self.sim.get_coords(
            region=region, it=it, exclude_ghosts=exclude_ghosts)
        it_dict = self.sim.its_lookup

        if (self.key in it_dict) and (region in it_dict[self.key]):
            return GridFunc(var=self,
                            region=region,
                            it=it,
                            coords=coords,
                            exclude_ghosts=exclude_ghosts)
        for ali in self.alias:
            if (ali in self.sim.its_lookup) and (region in self.sim.its_lookup[ali]):
                if self.sim.verbose > 1:
                    print(
                        f"{self.sim.sim_name}: Found alias key {ali} for {self.key}")
                self.key = ali
                return GridFunc(var=self,
                                region=region,
                                it=it,
                                coords=coords,
                                exclude_ghosts=exclude_ghosts)

        for bvar in self.backups:
            try:
                bu = bvar.get_data(region=region,
                                   it=it,
                                   exclude_ghosts=exclude_ghosts,
                                   **kwargs)
                if self.sim.verbose > 1:
                    print(
                        f"{self.sim.sim_name}: trying {bvar.key} instead of {self.key}")
                return bu
            except (VariableError, IterationError):
                continue
        raise IterationError(f"Iteration {it} not found for {self}")

    def available_its(self, region: str, **kwargs) -> 'NDArray[np.int_]':
        if (self.key in self.sim.its_lookup) and (region in self.sim.its_lookup[self.key]):
            return self.sim.its_lookup[self.key][region].astype(int)
        for ali in self.alias:
            if (ali in self.sim.its_lookup) and (region in self.sim.its_lookup[ali]):
                if self.sim.verbose > 1:
                    print(
                        f"{self.sim.sim_name}: Found alias key {ali} for {self.key}")
                self.key = ali
                return self.sim.its_lookup[ali][region]

        for bvar in self.backups:
            try:
                ret = bvar.available_its(region, **kwargs)
                if self.sim.verbose > 1:
                    print(
                        f"{self.sim.sim_name}: Using available its for {bvar.key} instead of {self.key}")
                return ret
            except (VariableError, IterationError):
                continue
        raise VariableError(f"{self} not in output for region {region}")


class TimeSeriesVariable(NativeVariable, TimeSeriesBaseVariable):
    """Variable for native time series functions"""

    def __init__(self, key: str, sim: "Simulation", ):
        super().__init__(key, sim, config.TimeSeriesVariables_json)

    def get_data(self, it=None, **kwargs):

        try:
            av_its = self.available_its(**kwargs)
            if it is None:
                it = av_its
            if len(uni := np.setdiff1d(it, av_its)) != 0:
                raise IterationError(
                    f"Iteration(s) {uni} not found for {self.key}")
            if isinstance(it, (int, np.integer)):
                return TimeSeries(self, its=np.array([it]), **kwargs).data[0]
            return TimeSeries(self, its=it, **kwargs)
        except (VariableError, IterationError):
            for ali in self.alias:
                if self.sim.verbose > 1:
                    print(
                        f"{self.sim.sim_name}: Found alias key {ali} for {self.key}")
                self.key = ali
                av_its = self.available_its(**kwargs)
                if it is None:
                    it = av_its
                if len(uni := np.setdiff1d(it, av_its)) != 0:
                    raise IterationError(
                        f"Iteration(s) {uni} not found for {self.key}")
                try:
                    if isinstance(it, (int, np.integer)):
                        return TimeSeries(self, its=np.array([it]), **kwargs).data[0]
                    return TimeSeries(self, its=it, **kwargs)
                except VariableError:
                    continue

            for bvar in self.backups:
                try:
                    av_its = bvar.available_its(**kwargs)
                    if it is None:
                        it = av_its
                    if len(uni := np.setdiff1d(it, av_its)) != 0:
                        raise IterationError(
                            f"Iteration(s) {uni} not found for {bvar.key}")
                    if isinstance(it, (int, np.integer)):
                        it = np.array([it])
                    if isinstance(bvar, PostProcVariable):
                        return PPTimeSeries(bvar, its=it, **kwargs)
                    else:
                        return TimeSeries(bvar, its=it, **kwargs)
                except (VariableError, IterationError):
                    continue

    def available_its(self, **kwargs) -> 'NDArray[np.int_]':
        try:
            return self.sim.data_handler.get_time_series(self.key)[0]
        except VariableError as ex:
            excp = ex
            for ali in self.alias:
                try:
                    its = self.sim.data_handler.get_time_series(ali)[0]
                    if self.sim.verbose > 1:
                        print(
                            f"{self.sim.sim_name}: Found alias key {ali} for {self.key}")
                    self.key = ali
                    return its
                except VariableError:
                    continue
            for bvar in self.backups:
                try:
                    ret = bvar.available_its(**kwargs)
                    if self.sim.verbose > 1:
                        print(
                            f"{self.sim.sim_name}: Using available its for {bvar.key} instead of {self.key}")
                    return ret
                except (VariableError, IterationError):
                    continue
        raise excp


class PPGridFuncVariable(PostProcVariable, GridFuncBaseVariable):
    """Variable for post processed grid functions"""

    def get_data(self,
                 region: str,
                 it: int,
                 exclude_ghosts: int = 0,
                 **kwargs) -> PPGridFunc:

        coords = self.sim.get_coords(
            region=region, it=it, exclude_ghosts=exclude_ghosts)
        kwargs = {**self.PPkeys, **kwargs}
        return PPGridFunc(var=self,
                          region=region,
                          it=it,
                          coords=coords,
                          exclude_ghosts=exclude_ghosts,
                          **kwargs)

    def available_its(self, region: str, **kwargs) -> 'NDArray[np.int_]':
        return super()._available_its(region, **kwargs)


class PPTimeSeriesVariable(PostProcVariable, TimeSeriesBaseVariable):
    """Variable for post processed time series functions"""
    reduction: Optional[Callable]

    def __init__(self, reduction: Optional[Callable] = None, **kwargs):
        super().__init__(**kwargs)
        self.reduction = reduction
        if any(isinstance(dep, (GridFuncVariable, PPGridFuncVariable)) for dep in self.dependencies):
            if self.reduction is None:
                raise ValueError("Post processed time series needs reduction operator "
                                 f"for GridFuncVariables in {self.dependencies}")

    def get_data(self, it=None, **kwargs):
        kwargs = {**self.PPkeys, **kwargs}
        try:
            av_its = self.available_its(**kwargs)
            if it is None:
                it = av_its
            if len(uni := np.setdiff1d(it, av_its)) != 0:
                raise IterationError(
                    f"Iteration(s) {uni} not found for {self}")
            if isinstance(it, (int, np.integer)):
                return PPTimeSeries(self, its=np.array([it]), **kwargs).data[0]
            return PPTimeSeries(self, its=it, **kwargs)
        except VariableError:
            for bvar in self.backups:
                try:
                    return bvar.get_data(it=it, **kwargs)
                except VariableError:
                    continue
            raise

    def available_its(self, **kwargs) -> 'NDArray[np.int_]':
        kwargs.setdefault('region', 'xz' if self.sim.is_cartoon else 'xyz')
        return super()._available_its(**kwargs)


class GravitationalWaveVariable(PostProcVariable, TimeSeriesBaseVariable):
    """Variable for post processed GW functions"""

    def __init__(self, **kwargs):
        super().__init__(dependencies=[], **kwargs)

    def get_data(self, it=None, **kwargs):
        return GWData(self, its=None, **kwargs)

    def available_its(self, **kwargs) -> 'NDArray[np.int_]':
        return self.get_data(**kwargs).its
