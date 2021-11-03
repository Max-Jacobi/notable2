from typing import Union, Optional, TYPE_CHECKING, Callable, Any
from collections.abc import Iterable, Mapping
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.contour import QuadContourSet
from matplotlib.image import AxesImage
from matplotlib.axes import Axes
from nptyping import NDArray

RLArgument = Optional[Union[int, Iterable]]

if TYPE_CHECKING:
    from .Simulation import Simulation
    from .Variable import Variable, GridFuncVariable, UGridFuncVariable, TimeSeriesVariable, UTimeSeriesVariable, PostProcVariable
    from .DataObjects import GridFunc, UGridFunc, TimeSeries, UTimeSeries


class PlotName():
    """String wrapper for plot labels"""
    unit: str
    code_unit: str
    format_options: list[str]

    def __init__(self,
                 name: str,
                 unit: str = "",
                 code_unit: Optional[str] = None,
                 format_options: Optional[list[str]] = None):
        self.name = name
        self.unit = unit
        self.code_unit = code_unit if code_unit is not None else self.unit
        self.format_options = format_options if format_options is not None else []

    def print(self, code_units=False, **kwargs):
        ret = self.name
        for key, repl in kwargs.items():
            if key not in self.format_options:
                continue
            ret = ret.replace(key, repl)
        if code_units:
            if self.code_unit != "":
                ret += f" [{self.code_unit}]"
        else:
            if self.unit != "":
                ret += f" [{self.unit}]"
        return ret

    def __str__(self):
        return self.name

    def __repr__(self):
        return f"PlotName: {self.name}"


Units = {"Rho": 6.175828477586656e+17,  # g/cm^3
         "Eps":  8.9875517873681764e+20,  # erg/g
         "Press":  5.550725674743868e+38,  # erg/cm^3
         "Mass": 1.988409870967742e+33,  # g
         "Energy": 1.7870936689836656e+3,  # 51 erg
         "Time": 0.004925490948309319,  # ms
         "Length":  1.4766250382504018}  # km
RUnits = {"Rho":  1.6192159539877191e-18,
          "Press":  1.8015662430410847e-39,
          "Eps":  1.1126500560536184e-21,
          "Mass":  5.028992139685286e-34,
          "Energy":  5.595508386114039e-55,
          "Time":  203.02544670054692,
          "Length":  0.6772199943086858}


class IterationError(Exception):
    ...


class VariableError(Exception):
    ...


class BackupException(Exception):
    backups: list["Variable"]

    def __init__(self, backups: list["Variable"]):
        super().__init__()
        self.backups = backups


def _save_log(dd):
    res = np.zeros_like(dd)*np.nan
    mask = dd > 0
    res[mask] = np.log10(dd[mask])
    return res


func_dict: dict[str, tuple[str, Callable]] = dict(
    log=('log({})', _save_log),
    logabs=('log($|${}$|$)', lambda d: np.log10(np.abs(d))),
)


class Plot2D(Mapping):

    rls: NDArray[np.int_]
    norm: Normalize
    first: Union[AxesImage, QuadContourSet]
    cmap: str
    axes: Axes
    kwarg: dict[str, Any]

    def __init__(self,
                 dictionary: dict[int, Union[AxesImage, QuadContourSet]],
                 norm: Normalize,
                 **kwargs):
        self._dict = dictionary
        self.rls = np.sort(list(self._dict.keys()))
        self.norm = norm
        rl = self.rls.min()
        self.first = self._dict[rl]
        self.cmap = self.first.get_cmap()
        self.axes = self.first.axes
        self.kwargs = kwargs

    def __getitem__(self, rl):
        return self._dict[rl]

    def __iter__(self):
        for rl in self.rls:
            yield rl

    def __len__(self):
        return len(self.rls)
