from typing import Any, Union
from abc import abstractmethod
from collections.abc import Mapping

from .DataGetters import DataGetter
from .Variable import Variable
from .Utils import NDArray


class GridDataBase(Mapping):
    """Documentation for GridData

    """

    var: Variable
    region: str
    it: int
    _data_getter: DataGetter
    time: float
    restart: int

    def __init__(self, var: Variable, region: str, it: int):
        self.var = var
        self.region = region
        self.it = it

        self.dim = len(region)
        self._grid_data_getter = self.var.sim.data_getter.get_grid_data
        self.time = self.var.sim.get_time(it=it)
        self.restart = self.var.sim.get_restart(it=it)

    @abstractmethod
    def __getitem__(self, rl: int) -> NDArray[Any, float]:
        ...

    def __iter__(self):
        for rl in self.var.sim.rls:
            yield rl

    def __len__(self):
        return len(self.var.sim.rls)


class GridData(GridDataBase):
    """Documentation for GridData

    """

    def __getitem__(self, rl):  # : int) -> NDArray[Any, float]:
        return self._grid_data_getter(self.var.key, rl, self.it, self.region)
