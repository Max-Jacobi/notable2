"""
Variable objects Inheritance tree:

- Variable
- UVariable

- GridVariable(Variable)
- UGridVariable(Variable)
- TimeSeriesVariable(Variable)
- UTimeSeries(Variable)
- ?Reduction seperate or as part of UTimeSeries?
"""

from abc import ABC, abstractmethod
from typing import Union, TYPE_CHECKING, Callable, Optional, Any
import json

from .RCParams import rcParams
from .DataObjects import GridData, UGridData, TimeSeries
from .Utils import PlotName, Units

if TYPE_CHECKING:
    from .Utils import Simulation


class Variable(ABC):
    """Documentation for Variable"""
    sim: "Simulation"
    key: str
    plot_name: PlotName
    kwargs: Optional[dict[str, Any]]
    scale_factor: float
    backups: list[str]

    def __init__(self, key: str, sim: "Simulation", ):
        self.sim = sim
        self.key = key
        self.backups = []

    def __repr__(self):
        return f"{self.__class__.__name__} {self.key}"

    def __str__(self):
        return f"{self.plot_name}"

    @abstractmethod
    def get_data(self, region: Optional[str] = None, it: Optional[int] = None, **kwargs) -> Union[TimeSeries, GridData]:
        ...


class GridVariable(Variable):
    """Variable for native grid functions"""

    file_key: str
    file_name: str

    def __init__(self, key: str, sim: "Simulation", ):
        super().__init__(key, sim)

        with open(rcParams.GridVariable_json, 'r') as ff:
            key_dict = json.load(ff)[key]
        self.file_key = key_dict.pop('file_key')
        self.file_name = key_dict.pop('file_name')
        self.scale_factor = key_dict.pop('scale_factor') if 'scale_factor' in key_dict else 1
        if isinstance(self.scale_factor, str):
            self.scale_factor = Units[self.scale_factor]
        self.kwargs = key_dict.pop('kwargs') if 'kwargs' in key_dict else {}

        self.plot_name = PlotName(**key_dict)

    def get_data(self, region: Optional[str] = None, it: Optional[int] = None, exclude_ghosts: int = 0, **kwargs) -> GridData:
        if region is None:
            region = 'xz'
        if it is None:
            it = 0
        return GridData(self, region, it, exclude_ghosts=exclude_ghosts)


class UGridVariable(Variable):
    """Variable for userdefined grid functions"""

    dependencies: list[Variable]

    def __init__(self,
                 key: str,
                 sim: "Simulation",
                 dependencies: list[str],
                 func: Callable,
                 plot_name_kwargs: dict[str, Any],
                 kwargs: dict[str, Any] = {},
                 ):

        super().__init__(key, sim)

        self.dependencies = [sim.get_variable(key) for key in dependencies]
        self.kwargs = kwargs
        self.func = func

        self.plot_name = PlotName(**plot_name_kwargs)

    def get_data(self, region: Optional[str] = None, it: Optional[int] = None, exclude_ghosts: int = 0, **kwargs) -> UGridData:
        if region is None:
            region = 'xz'
        if it is None:
            it = 0
        return UGridData(self, region, it, exclude_ghosts=exclude_ghosts, **kwargs)


class TimeSeriesVariable(Variable):
    """Variable for native time series functions"""

    file_key: str
    file_name: str

    def __init__(self, key: str, sim: "Simulation", ):
        super().__init__(key, sim)

        with open(rcParams.TimeSeriesVariables_json, 'r') as ff:
            key_dict = json.load(ff)[key]
        self.file_key = key_dict.pop('file_key')
        self.file_name = key_dict.pop('file_name')
        self.kwargs = key_dict.pop('kwargs') if 'kwargs' in key_dict else {}

        self.plot_name = PlotName(**key_dict)

    def get_data(self, region: Optional[str] = None, it: Optional[int] = None, **kwargs) -> TimeSeries:
        if it is None:
            return TimeSeries(self)
        ts = TimeSeries(self)
        return ts.data[ts.its == it][0]


class UTimeSeriesVariable(Variable):
    dependencies: list[Variable]
    ...


class VariableError(Exception):
    ...
