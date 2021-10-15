"""
Variable objects Inheritance tree:

- Variable
- UVariable

- GridDataVariable(Variable)
- UGridDataVariable(Variable)
- TimeSeriesVariable(Variable)
- UTimeSeries(Variable)
- ?Reduction seperate or as part of UTimeSeries?
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable, Optional, Any, overload
from functools import reduce
import json
import numpy as np

from .RCParams import rcParams
from .DataObjects import GridData, TimeSeries, UGridData, UTimeSeries
from .Utils import PlotName, Units, VariableError, BackupException, IterationError, NDArray

if TYPE_CHECKING:
    from .Utils import Simulation


class Variable(ABC):
    """Documentation for Variable"""
    sim: "Simulation"
    key: str
    vtype: str
    plot_name: PlotName
    kwargs: dict[str, Any]
    scale_factor: float
    backups: list["Variable"]
    get_data: Callable
    available_its: NDArray[np.int_]

    def __init__(self, key: str, sim: "Simulation", ):
        self.sim = sim
        self.key = key

    def __repr__(self):
        return f"{self.__class__.__name__} {self.key}"

    def __str__(self):
        return f"{self.plot_name}"


class TimeSeriesBaseVariable(ABC):
    """ABC for TimeSeriesVariables"""
    vtype: str = 'time'

    @overload
    def get_data(self, it: int, **kwargs) -> np.float_:
        ...

    @overload
    def get_data(self, it: Optional[NDArray[np.int_]], **kwargs) -> TimeSeries:
        ...

    @abstractmethod
    def get_data(self, it, **kwargs):
        ...

    @abstractmethod
    def available_its(self) -> NDArray[np.float_]:
        ...


class GridDataBaseVariable(ABC):
    """ABC for TimeSeriesVariables"""
    vtype: str = 'grid'

    @abstractmethod
    def get_data(self,
                 region: str,
                 it: int,
                 exclude_ghosts: int,
                 **kwargs) -> GridData:
        ...

    @abstractmethod
    def available_its(self, region: str) -> NDArray[np.float_]:
        ...


class NativeVariable(Variable):
    """ABC for Native variables"""
    file_key: str
    file_name: str

    def __init__(self, key: str, sim: "Simulation", json_path: str):
        super().__init__(key, sim)

        with open(json_path, 'r') as ff:
            try:
                key_dict = json.load(ff)[key]
            except KeyError as exc:
                raise VariableError(f"Key {key} not found in {json_path}") from exc

        self.file_key = key_dict.pop('file_key')
        self.file_name = key_dict.pop('file_name')
        self.scale_factor = key_dict.pop('scale_factor') if 'scale_factor' in key_dict else 1
        if isinstance(self.scale_factor, str):
            self.scale_factor = Units[self.scale_factor]
        self.kwargs = key_dict.pop('kwargs') if 'kwargs' in key_dict else {}
        self.backups = []
        if 'backups' in key_dict:
            for bu in key_dict.pop('backups'):
                try:
                    self.backups.append(self.sim.get_variable(bu))
                except VariableError:
                    continue

        self.plot_name = PlotName(**key_dict)

        if key not in sim.its_lookup:
            if any(self.backups):
                raise BackupException(self.backups)
            raise VariableError(f"Key {key} not in {sim}")


class UserVariable(Variable):
    """ABC for user defined Variables"""

    dependencies: NDArray[Variable]

    def __init__(self,
                 key: str,
                 sim: "Simulation",
                 dependencies: list[str],
                 func: Callable,
                 plot_name_kwargs: dict[str, Any],
                 backups: list[str] = [],
                 scale_factor: float = 1,
                 kwargs: dict[str, Any] = {},
                 ):

        super().__init__(key, sim)
        self.backups = []
        for bu in backups:
            try:
                self.backups.append(self.sim.get_variable(bu))
            except VariableError:
                continue

        self.kwargs = kwargs
        self.func = func
        self.scale_factor = scale_factor
        if isinstance(self.scale_factor, str):
            self.scale_factor = Units[self.scale_factor]
        self.plot_name = PlotName(**plot_name_kwargs)

        try:
            self.dependencies = np.array([sim.get_variable(key) for key in dependencies])
        except VariableError as exc:
            if any(self.backups):
                raise BackupException(self.backups) from exc
            raise exc


class GridDataVariable(NativeVariable, GridDataBaseVariable):
    """Variable for native grid functions"""

    def __init__(self, key: str, sim: "Simulation", ):
        super().__init__(key, sim, rcParams.GridDataVariable_json)

    def get_data(self,
                 region: str,
                 it: int,
                 exclude_ghosts: int = 0,
                 **kwargs) -> GridData:

        if it not in self.available_its(region=region):
            for bu_var in self.backups:
                try:
                    return bu_var.get_data(region=region, it=it, exclude_ghosts=exclude_ghosts, **kwargs)
                except (VariableError, IterationError):
                    continue
            else:
                raise IterationError(f"Iteration {it} not found for {self}")

        coords = self.sim.get_coords(region=region, it=it, exclude_ghosts=exclude_ghosts)

        return GridData(var=self,
                        region=region,
                        it=it,
                        coords=coords,
                        exclude_ghosts=exclude_ghosts)

    def available_its(self, region: str) -> NDArray[np.float_]:
        return self.sim.its_lookup[self.key][region]


class TimeSeriesVariable(NativeVariable, TimeSeriesBaseVariable):
    """Variable for native time series functions"""

    def __init__(self, key: str, sim: "Simulation", ):
        super().__init__(key, sim, rcParams.TimeSeriesVariables_json)

    def get_data(self, it=None, **kwargs):
        av_its = self.available_its()
        if it is None:
            it = av_its
        if len(uni := np.setdiff1d(it, av_its)) != 0:
            raise IterationError(f"Iteration(s) {uni} not found for self")
        if isinstance(it, int):
            return TimeSeries(self, its=np.array([it])).data[0]
        return TimeSeries(self, its=it)

    def available_its(self) -> NDArray[np.float_]:
        all_its, *_ = self.sim.data_handler.get_time_series(self.key)
        return all_its


class UGridDataVariable(UserVariable, GridDataBaseVariable):
    """Variable for user defined grid functions"""

    def __init__(self,
                 key: str,
                 sim: "Simulation",
                 dependencies: list[str],
                 func: Callable,
                 plot_name_kwargs: dict[str, Any],
                 scale_factor: float = 1,
                 kwargs: dict[str, Any] = {},
                 ):

        super().__init__(key=key,
                         sim=sim,
                         dependencies=dependencies,
                         func=func,
                         plot_name_kwargs=plot_name_kwargs,
                         scale_factor=scale_factor,
                         kwargs=kwargs)

    def get_data(self,
                 region: str,
                 it: int,
                 exclude_ghosts: int = 0,
                 **kwargs) -> UGridData:

        coords = self.sim.get_coords(region=region, it=it, exclude_ghosts=exclude_ghosts)
        return UGridData(var=self,
                         region=region,
                         it=it,
                         coords=coords,
                         exclude_ghosts=exclude_ghosts,
                         **kwargs)

    def available_its(self, region: str) -> NDArray[np.float_]:
        its = reduce(np.intersect1d, (dep.available_its(region) for dep in self.dependencies))
        if len(its) == 0:
            raise IterationError(f"No common iterations found for {self}")
        return its


class UTimeSeriesVariable(UserVariable, TimeSeriesBaseVariable):
    """Variable for user defined time series functions"""
    reduction: Optional[Callable]

    def __init__(self,
                 key: str,
                 sim: "Simulation",
                 dependencies: list[str],
                 func: Callable,
                 plot_name_kwargs: dict[str, Any],
                 scale_factor: float = 1,
                 reduction: Optional[Callable] = None,
                 kwargs: dict[str, Any] = {},
                 ):

        super().__init__(key=key,
                         sim=sim,
                         dependencies=dependencies,
                         func=func,
                         plot_name_kwargs=plot_name_kwargs,
                         scale_factor=scale_factor,
                         kwargs=kwargs)
        self.reduction = reduction
        if any(isinstance(dep, (GridDataVariable, UGridDataVariable)) for dep in self.dependencies):
            if self.reduction is None:
                raise ValueError("User-defined time series needs reduction operator "
                                 f"for GridDataVariables in {self.dependencies}")

    def get_data(self, it=None, **kwargs):
        av_its = self.available_its()
        if it is None:
            it = av_its
        if len(uni := np.setdiff1d(it, av_its)) != 0:
            raise IterationError(f"Iteration(s) {uni} not found for self")
        if isinstance(it, int):
            return UTimeSeries(self, its=np.array([it])).data[0]
        return UTimeSeries(self, its=it)

    def available_its(self) -> NDArray[np.float_]:
        region = 'xz' if self.sim.is_cartoon else 'xyz'
        its = reduce(np.intersect1d, (dep.available_its(region) for dep in self.dependencies))
        if len(its) == 0:
            raise IterationError(f"No common iterations found for {self}")
        return its
