"""
DataHandlers module

DataHandlers are responsible for parsing simulation output
and returning stuff like GridFunc and TimeSeries.
For a different simulation data layout implement a new DataHandler.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict
from pickle import loads
from gzip import decompress
from h5py import File  # type: ignore
import numpy as np

from .Utils import VariableError
if TYPE_CHECKING:
    from .Utils import Simulation
    from numpy.typing import NDArray


class DataHandler(ABC):
    """Abstract class for DataHandlers.
    DataHandlers are responsible for parsing simulation output
    and returning stuff like GridFunc and TimeSeries. """

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
    def get_grid_func(self, key: str, rl: int, it: int, region: str) -> 'NDArray[np.float_]':
        """Gets the GridFunc from simulation data"""
        ...

    @abstractmethod
    def get_time_series(self, key: str) -> 'NDArray[np.float_]':
        """Gets time series data from simulation data
        format: its, times, data, restarts"""
        ...


class PackETHandler(DataHandler):
    """Data getter for packETed data"""

    def __init__(self, sim: "Simulation"):
        super().__init__(sim)
        self.data = File(f'{self.sim.sim_path}/{self.sim.sim_name}.hdf5', 'r')
        self.structure = loads(decompress(self.data["structure"][()]), fix_imports=False)

    def get_structure(self, ):
        sdict = {}
        itdict = {}

        # set iterations, and setup times and restarts arrays
        itr = (np.array([int(kk) for kk in self.data.keys()
                         if kk not in ['structure', 'time_series']], dtype=int),
               np.zeros(n_its := len(self.data)-2, dtype=float),
               np.zeros(n_its, dtype=int))

        for ii, it in enumerate(itr[0]):
            # set times and restarts arrays
            dic, itr[1][ii], itr[2][ii] = self.structure[it]
            sDict[it] = {}

            # restructure dictionary formats
            for rl, dc in dic.items():
                sDict[it][rl] = {}
                for region, (keys, *(sDict[it][rl][region])) in dc.items():
                    for key in keys:
                        if key not in itdict.keys():
                            itDict[key] = {}
                        if region not in itDict[key].keys():
                            itDict[key][region] = []
                        itDict[key][region].append(it)

        # convert available iterations to a numpy array
        for dic in itdict.values():
            for rr, its in dic.items():
                dic[rr] = np.unique(its)

        # Time series for itdict
        for key, data in self.data['time_series'].items():
            itDict[key] = {'ts': data[0]}

        return itr, sdict, itdict

    def get_grid_func(self, key: str, rl: int, it: int, region: str) -> 'NDArray[np.float_]':
        dset_path = str(f'{it:08d}/{rl:02d}/{region}/{key}')
        dat = self.data[dset_path][()]
        dat[dat == 666] = np.nan
        return dat

    def get_time_series(self, key):
        try:
            return self.data[f'time_series/{key}'][...]
        except KeyError as excp:
            raise VariableError(f"{key} not found in simulation {self.sim}")


class PackET2Handler(DataHandler):
    """Data getter for packETed data"""

    def __init__(self, sim: "Simulation"):
        super().__init__(sim)
        self.data_dir = f'{self.sim.sim_path}/packET'
        self.structure_file = f'{self.data_dir}/structure.h5'

    def get_structure(self, ):
        sdict = {}
        itdict = {}

        with File(self.structure_file, 'r') as hf:
            # set iterations, times and restarts arrays
            itr = (hf['iterations'][:], hf['times'][:], hf['restarts'][:])

            # get coords
            for it_str in hf['coords']:
                it = int(it_str)
                sDict[it] = {}
                for rl_str in hf[f'coords/{it_str}']:
                    rl = int(rl_str)
                    sDict[it][rl] = {reg: (ccs[0], ccs[1], ccs[2].astype(int))
                                     for reg, ccs in hf[f'coords/{it_str}/{rl_str}'].items()}

            for key, reg_its in hf['available_its'].items():
                itDict[key] = {reg: its[:] for reg, its in reg_its.items()}

        return itr, sdict, itdict

    def get_grid_func(self, key: str, rl: int, it: int, region: str) -> 'NDArray[np.float_]':
        dset_path = f'{region}/{it:08d}/{rl:02d}'
        try:
            with File(f'{self.data_dir}/{key}.h5', 'r') as hf:
                dat = hf[dset_path][:]
            return dat
        except FileNotFoundError as excp:
            raise VariableError(f"{key} not found in simulation {self.sim}") from excp
        except OSError as excp:
            raise VariableError(f"{key} data is corrupted in simulation {self.sim}") from excp
        except KeyError as excp:
            raise VariableError(
                f"Data (it: {it}, rl: {rl}, region:{region}) "
                f"not found in data {key}.h5 of simulation {self.sim}"
            ) from excp

    def get_time_series(self, key):
        try:
            with File(f'{self.data_dir}/time_series.h5', 'r') as hf:
                dat = hf[key][:]
            return dat
        except KeyError as excp:
            raise VariableError(f"{key} not found in simulation {self.sim}") from excp
