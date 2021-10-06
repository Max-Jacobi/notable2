from typing import Any, TYPE_CHECKING
from abc import abstractmethod, ABC
from collections.abc import Mapping

from .DataHandlers import DataHandler
from .Utils import Units

if TYPE_CHECKING:
    from .Utils import NDArray, UGridVariable, Variable


class GridData(Mapping):
    """Documentation for GridData

    """

    var: "Variable"
    region: str
    it: int
    time: float
    restart: int
    exclude_ghosts: int

    def __init__(self, var: "Variable", region: str, it: int, exclude_ghosts: int = 0):
        self.var = var
        self.region = region
        self.it = it
        self.exclude_ghosts = exclude_ghosts

        self.dim = len(region)
        self.time = self.var.sim.get_time(it=it)
        self.restart = self.var.sim.get_restart(it=it)

    def __getitem__(self, rl):
        data = self.var.sim.data_handler.get_grid_data(self.var.key, rl, self.it, self.region)
        if self.exclude_ghosts != 0:
            data = data[self.exclude_ghosts:-self.exclude_ghosts]
            if self.dim > 1:
                data = data[:, self.exclude_ghosts:-self.exclude_ghosts]
            elif self.dim > 2:
                data = data[:, :, self.exclude_ghosts:-self.exclude_ghosts]
        return data

    def scaled(self, rl):
        return self[rl]*self.var.scale_factor

    def __iter__(self):
        for rl in self.var.sim.rls:
            yield self[rl]

    def __len__(self):
        return len(self.var.sim.rls)

    def __repr__(self):
        return f"GridData: {self.var.key}; {self.region}, it={self.it}, time={self.time*Units['Time']:.2f}ms"


class UGridData(GridData):
    """Documentation for UGridData"""

    def __init__(self, var: "UGridVariable", region: str, it: int, exclude_ghosts: int = 0, **kwargs):
        super().__init__(var, region, it, exclude_ghosts=exclude_ghosts)
        self.kwargs = kwargs

    def __getitem__(self, rl):
        dep_data = [dep.get_data(self.region, self.it, exclude_ghosts=self.exclude_ghosts, **self.kwargs)[rl]
                    for dep in self.var.dependencies]
        return self.var.func(*dep_data, **self.kwargs)


class TimeSeries():
    """Documentation for TimeSeries """

    var: "Variable"
    its: "NDArray[int]"
    data: "NDArray[float]"
    times: "NDArray[float]"
    restarts: "NDArray[int]"

    def __init__(self, var: "Variable"):
        self.var = var

        its, self.times, self.data, restarts = self.var.sim.data_handler.get_time_series(self.var.key)
        self.its = its.astype(int)
        self.restarts = restarts.astype(int)

    @property
    def scaled(self):
        return self.data * self.var.scale_factor

    def __len__(self):
        return len(self.its)

    def __repr__(self):
        return f"TimeSeries: {self.var.key}; time={self.times[0]*Units['Time']:.2f} - {self.times[-1]*Units['Time']:.2f}ms, length={len(self)}"
