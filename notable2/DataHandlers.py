"""
DataHandlers module

DataHandlers are responsible for parsing simulation output
and returning stuff like GridData and TimeSeries.
For a different simulation data layout implement a new DataHandler.
"""

from abc import ABC, abstractmethod
from typing import Any, TYPE_CHECKING
from pickle import loads
from gzip import decompress
from h5py import File  # type: ignore
import numpy as np

if TYPE_CHECKING:
    from .Utils import Simulation, NDArray


class DataHandler(ABC):
    """Abstract class for DataHandlers.
    DataHandlers are responsible for parsing simulation output
    and returning stuff like GridData and TimeSeries. """

    sim: "Simulation"

    def __init__(self, sim: "Simulation"):
        self.sim = sim

    @abstractmethod
    def get_structure(self, ):
        """
        Function to call at init.

        returns (its, times, restarts), structure_dict, iteration_lookup_dict

          structure_dict keypattern: it -> rl -> region -> (origins, dxs, Ns)
          iteration_lookup_dict keypattern: variable key -> region -> available its
        """
        ...

    @abstractmethod
    def get_grid_data(self, key: str, rl: int, it: int, region: str) -> "NDArray[Any, float]":
        """Gets the GridData from simulation data"""
        ...

    @abstractmethod
    def get_time_series(self, key: str) -> "NDArray[(4, Any), float]":
        """Gets the GridData from simulation data"""
        ...


class PackETGetter(DataHandler):
    """Data getter for packETed data"""

    def __init__(self, sim: "Simulation"):
        super().__init__(sim)
        self.data = File(f'{self.sim.sim_path}/{self.sim.sim_name}.hdf5', 'r')
        self.structure = loads(decompress(self.data["structure"][()]), fix_imports=False)

    def get_structure(self, ):
        sdict = {}
        itdict = {}

        # set iterations, and setup times and restarts arrays
        itr = (np.array([kk for kk in self.data.keys()
                         if kk not in ['structure', 'time_series']], dtype=int),
               np.zeros(n_its := len(self.data), dtype=float),
               np.zeros(n_its, dtype=int))

        for ii, it in enumerate(itr[0]):
            # set times and restarts arrays
            dic, itr[1][ii], itr[2][ii] = self.structure[it]
            sdict[it] = {}

            # restructure dictionary formats
            for rl, dc in dic.items():
                sdict[it][rl] = {}
                for region, (keys, *(sdict[it][rl][region])) in dc.items():
                    for key in keys:
                        if key not in itdict.keys():
                            itdict[key] = {}
                        if region not in itdict[key].keys():
                            itdict[key][region] = []
                        itdict[key][region].append(it)

        # Put available iterations in a numpy array
        for dic in itdict.values():
            for rr, its in dic.items():
                dic[rr] = np.array(its)

        return itr, sdict, itdict

    def get_grid_data(self, key: str, rl: int, it: int, region: str) -> "NDArray[Any, float]":
        dat = self.data[f'{it:08d}/{rl:02d}/{region}/{key}'][()]
        dat[dat == 666] = np.nan
        return dat

    def get_time_series(self, key):
        return self.data[f'time_series/{key}'][...]
