"""
Thi is the DataHandlers module.

DataHandlers are responsible for parsing simulation output
and returning stuff like GridFunc and TimeSeries.
For a different simulation data layout implement a new DataHandler.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any
from pickle import loads
from gzip import decompress
import os
import h5py as h5
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
    def get_structure(self, ) -> tuple:
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


class OldPackETHandler(DataHandler):
    """Data getter for packETed data"""

    def __init__(self, sim: "Simulation"):
        super().__init__(sim)
        self.data = h5.File(f'{self.sim.sim_path}/{self.sim.sim_name}.hdf5', 'r')
        self.structure = loads(decompress(
            self.data["structure"][()]), fix_imports=False)

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

        # convert available iterations to a numpy array
        for dic in itdict.values():
            for rr, its in dic.items():
                dic[rr] = np.unique(its)

        # Time series for itdict
        for key, data in self.data['time_series'].items():
            itdict[key] = {'ts': data[0]}

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


class PackETHandler(DataHandler):
    """Data getter for packETed data"""

    def __init__(self, sim: "Simulation"):
        super().__init__(sim)
        self.data_dir = f'{self.sim.sim_path}/packET_data'

    def get_structure(self, ):
        if os.path.isfile(f"{self.data_dir}/structure.h5"):
            with h5.File(f"{self.data_dir}/structure.h5", 'r') as hf:
                itr = tuple(np.array(hf[key][:]) for key in ('its', 'times', 'restarts'))
                sdict = _hdf5_to_dict(hf['structure'])
                itdict = _hdf5_to_dict(hf['available_its'])
            return itr, sdict, itdict

        sdict = {}
        itdict = {}
        iterations, times = [], []

        for reg in "x y z xy xz yz xyz".split():
            for h5filename in os.listdir(f"{self.data_dir}/{reg}"):
                key = h5filename.split(".")[0]
                with h5.File(f"{self.data_dir}/{reg}/{h5filename}", 'r') as hf:
                    its = np.array([int(it_str) for it_str in hf])
                    for it_str in hf.keys():
                        it = int(it_str)
                        iterations.append(it)
                        times.append(hf[it_str].attrs['time'])
                        if it not in sdict:
                            sdict[it] = {}
                        for rl_str in hf[it_str].keys():
                            rl = int(rl_str)
                            if rl not in sdict[it]:
                                sdict[it][rl] = {}
                            if reg in sdict[it][rl]:
                                continue
                            orig = np.array(hf[f"{it_str}/{rl_str}"].attrs['origin'][:])
                            dx = np.array(hf[f"{it_str}/{rl_str}"].attrs['delta'][:])
                            nn = np.array(hf[f"{it_str}/{rl_str}"].shape)
                            sdict[it][rl][reg] = (orig, dx, nn)
                if key not in itdict:
                    itdict[key] = {}
                itdict[key][reg] = its

        iterations = np.array(iterations)
        times = np.array(times)
        iterations, ii = np.unique(iterations, return_index=True)
        times = times[ii]
        restarts = np.zeros_like(iterations)
        with h5.File(f"{self.data_dir}/structure.h5", 'w') as hf:
            hf['its'] = iterations
            hf['times'] = times
            hf['restarts'] = restarts

            _dict_to_hdf5(hf.create_group('structure'), sdict)
            _dict_to_hdf5(hf.create_group('available_its'), itdict)

        return (iterations, times, restarts), sdict, itdict


    def get_grid_func(self, key: str, rl: int, it: int, region: str) -> 'NDArray[np.float_]':
        dset_path = f'{it:010d}/{rl:02d}'
        try:
            with h5.File(f'{self.data_dir}/{region}/{key}.hdf5', 'r') as hf:
                dat = hf[dset_path][:]
            return dat
        except FileNotFoundError as excp:
            raise VariableError(
                f"{key} not found in simulation {self.sim}") from excp
        except OSError as excp:
            raise VariableError(
                f"{key} data is corrupted in simulation {self.sim}") from excp
        except KeyError as excp:
            raise VariableError(
                f"Data (it: {it}, rl: {rl}, region:{region}) "
                f"not found in data {key}.h5 of simulation {self.sim}"
            ) from excp

    def get_time_series(self, key):
        try:
            with h5.File(f'{self.data_dir}/time_series/{key}.hdf5', 'r') as hf:
                it, time, dat = hf[key][:]
            return it, time, dat, np.zeros_like(it)
        except KeyError as excp:
            raise VariableError(
                f"{key} not found in simulation {self.sim}") from excp

class PackET2Handler(DataHandler):
    """Data getter for packETed data"""

    def __init__(self, sim: "Simulation"):
        super().__init__(sim)
        self.data_dir = f'{self.sim.sim_path}/packET'
        self.structure_file = f'{self.data_dir}/structure.h5'

    def get_structure(self, ):
        sdict = {}
        itdict = {}

        with h5.File(self.structure_file, 'r') as hf:
            # set iterations, times and restarts arrays
            itr = (hf['iterations'][:], hf['times'][:], hf['restarts'][:])

            # get coords
            for it_str in hf['coords']:
                it = int(it_str)
                sdict[it] = {}
                for rl_str in hf[f'coords/{it_str}']:
                    rl = int(rl_str)
                    sdict[it][rl] = {reg: (ccs[0], ccs[1], ccs[2].astype(int))
                                     for reg, ccs in hf[f'coords/{it_str}/{rl_str}'].items()}

            for key, reg_its in hf['available_its'].items():
                itdict[key] = {reg: its[:] for reg, its in reg_its.items()}

        return itr, sdict, itdict

    def get_grid_func(self, key: str, rl: int, it: int, region: str) -> 'NDArray[np.float_]':
        dset_path = f'{region}/{it:08d}/{rl:02d}'
        try:
            with h5.File(f'{self.data_dir}/{key}.h5', 'r') as hf:
                dat = hf[dset_path][:]
            return dat
        except FileNotFoundError as excp:
            raise VariableError(
                f"{key} not found in simulation {self.sim}") from excp
        except OSError as excp:
            raise VariableError(
                f"{key} data is corrupted in simulation {self.sim}") from excp
        except KeyError as excp:
            raise VariableError(
                f"Data (it: {it}, rl: {rl}, region:{region}) "
                f"not found in data {key}.h5 of simulation {self.sim}"
            ) from excp

    def get_time_series(self, key):
        try:
            with h5.File(f'{self.data_dir}/time_series.h5', 'r') as hf:
                dat = hf[key][:]
            return dat
        except KeyError as excp:
            raise VariableError(
                f"{key} not found in simulation {self.sim}") from excp

def _hdf5_to_dict(h5_obj):
    result = {}
    for key, item in h5_obj.items():
        try:
            key = int(key)
        except ValueError:
            pass
        if isinstance(item, h5.Dataset):
            result[key] = item[()]  # efficiently extract dataset content
        elif isinstance(item, h5.Group):
            result[key] = _hdf5_to_dict(item)  # recursively handle groups
    return result

def _dict_to_hdf5(h5_obj, data_dict):
    for key, item in data_dict.items():
        key = str(key)
        if isinstance(item, dict):
            group = h5_obj.create_group(key)
            _dict_to_hdf5(group, item)  # Recursive call for nested dictionaries
        else:
            h5_obj.create_dataset(key, data=item)
