from abc import ABC, abstractmethod
from typing import Callable, Optional, TYPE_CHECKING, List, Dict, Tuple
from functools import reduce
import numpy as np
from h5py import File  # type: ignore
import alpyne.uniform_interpolation as ui  # type: ignore

from .Utils import RUnits

if TYPE_CHECKING:
    from numpy.typing import NDArray


class EOS(ABC):
    """EOS abstract base class"""

    @abstractmethod
    def get_caller(self,
                   keys: List[str],
                   func: Callable = lambda *args: args[0]
                   ) -> Callable:
        """
        Returns a callable for creation of post processed data.
        func is called with the signature (*data, rho, temp, ye),
        where data are the arrays generated from keys
        """
        ...

    @abstractmethod
    def get_cold_caller(self,
                        keys: List[str],
                        func: Callable = lambda *args: args[0]
                        ) -> Callable:
        """
        Returns a callable for creation of post processed data.
        func is called with the signature (*data, rho, ye),
        where data are the arrays generated from keys
        """
        ...

    @abstractmethod
    def get_inf_caller(self,
                       keys: List[str],
                       func: Callable = lambda *args: args[0]
                       ) -> Callable:
        """
        Returns a callable for creation of post processed data.
        func is called with the signature (*data, ye),
        where data are the arrays generated from keys
        """
        ...


class TabulatedEOS(EOS):
    """Realistic Tabluated EOS """
    hydro_path: str
    weak_path: str
    data: Dict[str, 'NDArray[np.float_]']

    def __init__(self, path: str):
        self._table: Optional[List['NDArray[np.float_]']] = None
        self._table_cold: Optional[List['NDArray[np.float_]']] = None
        self._ye_r: Optional[Tuple[np.float_]] = None
        self._temp_r: Optional[Tuple[np.float_]] = None
        self._rho_r: Optional[Tuple[np.float_]] = None
        self.data = {}
        self.set_path(path)
        self.name = path.split('/')[-1]

    def __str__(self) -> str:
        return self.hydro_path.replace('hydro.h5', '')

    @property
    def ye_range(self) -> Tuple[np.float_]:
        if self._ye_r is None:
            with File(self.hydro_path, 'r') as hfile:
                Ye = np.array(hfile['ye'])
            self._ye_r = (Ye[0], Ye[-1])
        return self._ye_r

    @property
    def temp_range(self) -> Tuple[np.float_]:
        if self._temp_r is None:
            with File(self.hydro_path, 'r') as hfile:
                Temp = np.array(hfile['temperature'])
            self._temp_r = (Temp[0], Temp[-1])
        return self._temp_r

    @property
    def rho_range(self) -> Tuple[np.float_]:
        if self._rho_r is None:
            with File(self.hydro_path, 'r') as hfile:
                Rho = np.array(hfile['density'])*RUnits['Rho']
            self._rho_r = (Rho[0], Rho[-1])
        return self._rho_r

    @property
    def table(self) -> List['NDArray[np.float_]']:
        if self._table is None:
            with File(self.hydro_path, 'r') as hfile:
                Ye = np.array(hfile['ye'])
                ltemp = np.log10(hfile['temperature'])
                lrho = np.log10(hfile['density']) + np.log10(RUnits['Rho'])
                iye = 1/(Ye[1]-Ye[0])
                iltemp = 1/(ltemp[1]-ltemp[0])
                ilrho = 1/(lrho[1]-lrho[0])
                self._table = [np.array([Ye[0], ltemp[0], lrho[0]]),
                               np.array([iye, iltemp, ilrho])]
        return self._table

    def get_key(self, key):
        self._get_keys([key])
        return self.data[key]

    def get_inf_caller(self,
                       keys: List[str],
                       func: Callable = lambda *args: args[0]
                       ) -> Callable:
        def eos_caller_inf(ye: 'NDArray[np.float_]',
                           *_, **kw) -> 'NDArray[np.float_]':

            nonlocal keys
            nonlocal func

            self._get_keys(keys)
            scalars = [kk for kk in keys if np.isscalar(self.data[kk])]

            shape = ye.shape
            fshape = (np.prod(shape), )

            ye = ye.flatten()
            mask = np.isfinite(ye)
            ye = ye[mask]

            data = np.array([self.data[kk][:, 0, 0] for kk in keys
                             if kk not in scalars])
            islog = np.array([np.all(dd > 0) for dd in data])
            data[islog] = np.log10(data[islog])

            res = ui.linterp1D(ye,
                               self.table[0][0],
                               self.table[1][0],
                               data)

            args = []
            for kk in keys:
                i_int = 0
                if kk in scalars:
                    tmp = self.data[kk]
                else:
                    tmp = np.zeros(fshape)*np.nan
                    tmp[mask] = 10**res[i_int] if islog[i_int] else res[i_int]
                    tmp = np.reshape(tmp, shape)
                    i_int += 1
                args.append(tmp)

            return func(*args, ye, **kw)

        return eos_caller_inf

    def get_cold_caller(self,
                        keys: List[str],
                        func: Callable = lambda *args: args[0]
                        ) -> Callable:
        def eos_caller_cold(ye: 'NDArray[np.float_]',
                            rho: 'NDArray[np.float_]',
                            *_, **kw) -> 'NDArray[np.float_]':

            nonlocal keys
            nonlocal func

            self._get_keys(keys)
            scalars = [kk for kk in keys if np.isscalar(self.data[kk])]

            shape = ye.shape
            fshape = (np.prod(shape), )

            args = [ye.flatten(), np.log10(rho).flatten()]
            mask = reduce(np.logical_and, [np.isfinite(arg) for arg in args])
            args = [arg[mask] for arg in args]

            data = np.array([self.data[kk][:, 0] for kk in keys
                             if kk not in scalars])
            islog = np.array([np.all(dd > 0) for dd in data])
            data[islog] = np.log10(data[islog])

            res = ui.linterp2D(*args,
                               self.table[0][[0, 2]],
                               self.table[1][[0, 2]],
                               data)

            args = []
            for kk in keys:
                i_int = 0
                if kk in scalars:
                    tmp = self.data[kk]
                else:
                    tmp = np.zeros(fshape)*np.nan
                    tmp[mask] = 10**res[i_int] if islog[i_int] else res[i_int]
                    tmp = np.reshape(tmp, shape)
                    i_int += 1
                args.append(tmp)

            return func(*args, ye, **kw)

        return eos_caller_cold

    def get_caller(self,
                   keys: List[str],
                   func: Callable = lambda *args: args[0]
                   ) -> Callable:
        def eos_caller(ye: 'NDArray[np.float_]',
                       temp: 'NDArray[np.float_]',
                       rho: 'NDArray[np.float_]',
                       *_, **kw) -> 'NDArray[np.float_]':

            nonlocal keys
            nonlocal func

            self._get_keys(keys)
            scalars = [kk for kk in keys if np.isscalar(self.data[kk])]

            shape = ye.shape
            fshape = (np.prod(shape), )

            args = [ye.flatten(), np.log10(temp).flatten(),
                    np.log10(rho).flatten()]
            mask = reduce(np.logical_and, [np.isfinite(arg) for arg in args])
            args = [arg[mask] for arg in args]

            data = np.array([self.data[kk] for kk in keys])
            islog = np.array([np.all(dd > 0) for dd in data])
            data[islog] = np.log10(data[islog])

            res = ui.linterp3D(*args, *self.table, data)

            args = []
            for kk in keys:
                i_int = 0
                if kk in scalars:
                    tmp = self.data[kk]
                else:
                    tmp = np.zeros(fshape)*np.nan
                    tmp[mask] = 10**res[i_int] if islog[i_int] else res[i_int]
                    tmp = np.reshape(tmp, shape)
                    i_int += 1
                args.append(tmp)

            return func(*args, ye, **kw)

        return eos_caller

    def _get_keys(self, keys: List[str]):
        if self.hydro_path is None:
            raise OSError(
                "Path to EOS file not given. "
                "Run Simulation.eos.set_path('path/to/eos/')")

        new_keys = [kk for kk in keys if kk not in self.data]

        _scale = dict(
            pressure=RUnits["Press"],
            only_P=RUnits["Press"],
            internalEnergy=RUnits["Eps"],
            only_E=RUnits["Eps"],
            density=RUnits["Rho"],
        )

        if len(new_keys) > 0:
            for kk in new_keys:
                for path in [self.hydro_path, self.weak_path]:
                    with File(path, 'r') as hfile:
                        if kk in hfile:
                            self.data[kk] = hfile[kk][()]
                            if kk in _scale:
                                self.data[kk] *= _scale[kk]
                            break
                else:
                    raise KeyError(f"{kk} not found in EOS tables in {self}")

    def set_path(self, path: str):
        "EOS path setter"
        self.hydro_path = f"{path}/hydro.h5"
        self.weak_path = f"{path}/weak.h5"
