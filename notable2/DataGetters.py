from abc import ABC, abstractmethod
from typing import Any
from pickle import loads
from gzip import decompress
from h5py import File
import numpy as np

from .Utils import Simulation, NDArray, GridStructure


class DataGetter(ABC):
    """Abstract class for DataGetters.
    DataGetters are responsible for parsing simulation output and returning stuff like GridData and TimeSeries.

    """
    sim: Simulation

    def __init__(self, sim: Simulation):
        self.sim = sim

    @abstractmethod
    def get_structure(self, ) -> GridStructure:
        ...

    @abstractmethod
    def get_grid_data(self, key: str, rl: int, it: int, region: str) -> NDArray[Any, float]:
        ...

    @abstractmethod
    def get_time_series(self, key: str) -> NDArray[(4, Any), float]:
        ...


class packETGetter(DataGetter):
    """Data getter for packETed data"""

    def __init__(self, sim: Simulation):
        super().__init__(sim)
        self.data = File(f'{self.sim.sim_path}/{self.sim.sim_name}.hdf5', 'r')

        self.its = np.array(list(self.data.keys()))

    def get_structure(self, ):
        """Gets the reformatted structure dict from the hdf5"""
        sdict = {}
        struc = loads(decompress(self.data["structure"][()]), fix_imports=False)
        times = np.zeros_like(self.its)
        restarts = np.zeros_like(self.its)
        for ii, it in enumerate(self.its):
            dic, tt, rr = struc[it]
            times[ii] = tt
            restarts[ii] = rr
            for rl, dc in dic.items():
                for region, (keys, *grid_data) in dc.items():
                    for key in keys:
                        if key not in sdict.keys():
                            sdict[key] = {}
                        if region not in sdict[key].keys():
                            sdict[key][region] = {}
                        sdict[key][region][rl] = grid_data
        return GridStructure(sdict)

    def get_grid_data(self, key: str, rl: int, it: int, region: str) -> NDArray[Any, float]:
        """Gets the GridData from packET hdf5 file"""
        dat = self.data[f'{it:08d}/{rl:02d}/{region}/{key}']
        dat[dat == 666] = np.nan
        return dat

    def get_time_series(self, key):
        """Gets the GridData from packET hdf5 file"""
        return self.data[f'time_series/{key}'][...]
