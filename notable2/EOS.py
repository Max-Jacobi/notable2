from abc import ABC, abstractmethod
from typing import Callable, Optional
from functools import reduce
import numpy as np
from numpy.typing import NDArray
from h5py import File  # type: ignore
import alpyne.uniform_interpolation as ui  # type: ignore


class EOS(ABC):
    """EOS abstract base class"""

    @abstractmethod
    def get_caller(self,
                   keys: list[str],
                   func: Callable = lambda *args: args[0]
                   ) -> Callable:
        """
        Returns a callable for creation of user defined data.
        func is called with the signature (*data, rho, temp, ye),
        where data are the arrays generated from keys
        """
        ...

    @abstractmethod
    def get_cold_caller(self,
                        keys: list[str],
                        func: Callable = lambda *args: args[0]
                        ) -> Callable:
        """
        Returns a callable for creation of user defined data.
        func is called with the signature (*data, rho, ye),
        where data are the arrays generated from keys
        """
        ...


class TabulatedEOS(EOS):
    """Realistic Tabluated EOS """
    path: str
    data: dict[str, NDArray[np.float_]]

    def __init__(self, path: str):
        self.path = path
        self._table: Optional[list[NDArray[np.float_]]] = None
        self._table_cold: Optional[list[NDArray[np.float_]]] = None
        self.data = {}

    @property
    def table(self) -> list[NDArray[np.float_]]:
        if self._table is None:
            with File(self.path, 'r') as hfile:
                Ye = np.array(hfile['ye'])
                ltemp = np.log10(hfile['temperature'])
                lrho = np.log10(hfile['density']) + np.log10(1.6192159539877191e-18)
                iye = 1/(Ye[1]-Ye[0])
                iltemp = 1/(ltemp[1]-ltemp[0])
                ilrho = 1/(lrho[1]-lrho[0])
            self._table = [np.array([Ye[0], ltemp[0], lrho[0]]),
                           np.array([iye, iltemp, ilrho])]
        return self._table

    @property
    def table_cold(self) -> list[NDArray[np.float_]]:
        if self._table_cold is None:
            with File(self.path, 'r') as hfile:
                Ye = np.array(hfile['ye'])
                lrho = np.log10(hfile['density']) + np.log10(1.6192159539877191e-18)
                iye = 1/(Ye[1]-Ye[0])
                ilrho = 1/(lrho[1]-lrho[0])
            self._table_cold = [np.array([Ye[0], lrho[0]]),
                                np.array([iye, ilrho])]
        return self._table_cold

    def get_cold_caller(self,
                        keys: list[str],
                        func: Callable = lambda *args: args[0]
                        ) -> Callable:
        def eos_caller_cold(ye: NDArray[np.float_],
                            rho: NDArray[np.float_],
                            *_, **kw) -> NDArray[np.float_]:

            nonlocal keys
            nonlocal func

            self._get_keys(keys)

            shape = ye.shape
            fshape = (np.prod(shape), )

            args = [ye.flatten(), np.log10(rho).flatten()]
            mask = reduce(np.logical_and, [np.isfinite(arg) for arg in args])
            args = [arg[mask] for arg in args]

            res = ui.linterp2D(*args, *self.table, [np.log10(self.data[kk]) for kk in keys])

            data = [np.zeros(fshape)*np.nan for _ in keys]
            for dd, rr in zip(data, res):
                dd[mask] = 10**rr
            data = [np.reshape(dd, shape) for dd in data]

            return func(*data, rho, ye, **kw)

        return eos_caller_cold

    def get_caller(self,
                   keys: list[str],
                   func: Callable = lambda *args: args[0]
                   ) -> Callable:
        def eos_caller(ye: NDArray[np.float_],
                       temp: NDArray[np.float_],
                       rho: NDArray[np.float_],
                       *_, **kw) -> NDArray[np.float_]:

            nonlocal keys
            nonlocal func

            self._get_keys(keys)

            shape = ye.shape
            fshape = (np.prod(shape), )

            args = [ye.flatten(), np.log10(temp).flatten(), np.log10(rho).flatten()]
            mask = reduce(np.logical_and, [np.isfinite(arg) for arg in args])
            args = [arg[mask] for arg in args]

            res = ui.linterp3D(*args, *self.table, [np.log10(self.data[kk]) for kk in keys])

            data = [np.zeros(fshape)*np.nan for _ in keys]
            for dd, rr in zip(data, res):
                dd[mask] = 10**rr
            data = [np.reshape(dd, shape) for dd in data]

            return func(*data, rho, temp, ye, **kw)

        return eos_caller

    def _get_keys(self, keys: list[str]):
        if self.path is None:
            raise OSError("Path to EOS file not given. Run Simulation.eos.set_path('path/to/eos/hydro.h5'")

        new_keys = [kk for kk in keys if kk not in self.data.keys()]

        if len(new_keys) > 0:
            with File(self.path, 'r') as hfile:
                for kk in new_keys:
                    self.data[kk] = hfile[kk][()]

    def set_path(self, path: str):
        "EOS path setter"
        self.path = path
