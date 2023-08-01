import os
from typing import TYPE_CHECKING, Any, Optional, Sequence, Dict
from inspect import signature
from collections.abc import Mapping
from h5py import File as HDF5  # type: ignore
import numpy as np
from alpyne.uniform_interpolation import (linterp1D,  linterp2D,  # type: ignore
                                          linterp3D,  chinterp1D,  # type: ignore
                                          chinterp2D,  chinterp3D)  # type: ignore

from .Utils import Units, VariableError

if TYPE_CHECKING:
    from .Utils import PPTimeSeriesVariable, PPGridFuncVariable, Variable
    from numpy.typing import NDArray


class GridFunc(Mapping):
    """Documentation for GridFunc

    """

    var: "Variable"
    coords: Dict[int, Dict[str, 'NDArray[np.float_]']]
    region: str
    it: int
    time: float
    restart: int
    exclude_ghosts: int
    mem_data: Dict[int, 'NDArray[np.float_]']

    def __init__(self,
                 var: "Variable",
                 region: str,
                 it: int,
                 coords: Dict[int, Dict[str, 'NDArray[np.float_]']],
                 exclude_ghosts: int = 0):
        self.var = var
        self.region = region
        self.it = it
        self.exclude_ghosts = exclude_ghosts
        self.coords = coords

        self.dim = len(region)
        self.time = self.var.sim.get_time(it=it)
        self.restart = self.var.sim.get_restart(it=it)
        self.mem_load = False
        self.mem_data = {}

    def __getitem__(self, rl):
        if rl in self.mem_data:
            return self.mem_data[rl]

        try:
            data = self.var.sim.data_handler.get_grid_func(
                self.var.key, rl, self.it, self.region)
        except VariableError:
            if rl > 0:
                return self.__getitem__(rl-1)
            else:
                raise

        if self.exclude_ghosts != 0:
            data = data[self.exclude_ghosts:-self.exclude_ghosts]
            if self.dim > 1:
                data = data[:, self.exclude_ghosts:-self.exclude_ghosts]
            elif self.dim > 2:
                data = data[:, :, self.exclude_ghosts:-self.exclude_ghosts]
        if self.mem_load:
            self.mem_data[rl] = data
        return data

    def __call__(self, kind: str = 'linear',
                 **int_coords: 'NDArray[np.float_]'
                 ) -> 'NDArray[np.float_]':

        if len(int_coords) == 0:
            raise ValueError("No interpolation points give")
        if len(missing := [ax for ax in int_coords
                           if ax not in self.region]) != 0:
            raise ValueError(f"Axes {missing} not in GridFunc {self}")

        for ax, cc in int_coords.items():
            if isinstance(cc, (int, float, np.number)):
                int_coords[ax] = np.array([cc])
            else:
                int_coords[ax] = np.array(cc)

        result = 666.*np.ones_like(int_coords[ax])

        if kind == 'linear':
            if self.dim == 1:
                interpolate = linterp1D
            elif self.dim == 2:
                interpolate = linterp2D
            elif self.dim == 3:
                interpolate = linterp3D
            loff, roff = 0, -1
        elif kind == 'cubic':
            if self.dim == 1:
                interpolate = chinterp1D
            elif self.dim == 2:
                interpolate = chinterp2D
            elif self.dim == 3:
                interpolate = chinterp3D
            loff, roff = 1, -2
        else:
            raise ValueError(f"{kind} kind interpolation not supported")

        for rl in self:
            coords_level = self.coords[rl]
            dx = np.array([cc[1]-cc[0] for cc in coords_level.values()])
            ih = 1/dx
            orig = np.array([cc[0] for cc in coords_level.values()])
            mask = np.ones_like(result).astype(bool)
            for ax in int_coords:
                cl = coords_level[ax]
                cc = int_coords[ax]
                mask = mask & ((cl[loff] <= cc) & (cl[roff] > cc))

            if not np.any(mask):
                continue
            interp, = interpolate(*[int_coords[ax][mask] for ax in coords_level],
                                  orig, ih, np.array([self[rl]]))

            result[mask] = interp
            assert not np.any(interp == 666.), (
                f"interpolation error on rl {rl}\n"
                f"{{ax: cc[mask][np.isnan(interp) for ax, cc in int_coords.items()]}}"
            )
        missing_mask = result == 666.
        if np.any(missing_mask):
            result[missing_mask] = np.nan
        return result

    def __iter__(self):
        for rl in self.var.sim.rls[self.it]:
            yield rl

    def __len__(self):
        return len(self.var.sim.rls[self.it])

    def __str__(self):
        return f"{self.var.key}; {self.region}, it={self.it}, time={self.time*Units['Time']:.2f}ms"

    def __repr__(self):
        return f"GridFunc: {self.var.key}; {self.region}, it={self.it}, time={self.time*Units['Time']:.2f}ms"

    def scaled(self, rl):
        return self[rl]*self.var.scale_factor


class PPGridFunc(GridFunc):
    """Documentation for PPGridFunc"""
    kwargs: Dict[str, Any]

    def __init__(self,
                 var: "PPGridFuncVariable",
                 region: str,
                 it: int,
                 coords: Dict[int, Dict[str, 'NDArray[np.float_]']],
                 exclude_ghosts: int = 0,
                 **kwargs):
        super().__init__(var=var, region=region, it=it, coords=coords,
                         exclude_ghosts=exclude_ghosts)
        self.kwargs = kwargs

    def __getitem__(self, rl):

        if rl in self.mem_data:
            return self.mem_data[rl]

        # checked for saved version
        if self.var.save:
            key = self.var.key
            for kk, item in self.kwargs.items():
                if kk in self.var.PPkeys:
                    key += f":{kk}={item}"
            dset_path = f'{self.region}/{self.it:08d}/{rl}'
            if os.path.isfile(f'{self.var.sim.pp_hdf5_path}/{key}.h5'):
                with HDF5(f'{self.var.sim.pp_hdf5_path}/{key}.h5', 'r') as hdf5:
                    if dset_path in hdf5:
                        data = np.array(hdf5[dset_path])
                        if self.mem_load:
                            self.mem_data[rl] = data
                        return data

        # get dependencies
        dep_data = [dep.get_data(region=self.region,
                                 it=self.it,
                                 exclude_ghosts=self.exclude_ghosts,
                                 **self.kwargs)
                    for dep in self.var.dependencies]

        # get correct refinementlevel of grid data
        for ii, dep in enumerate(self.var.dependencies):
            if dep.vtype == 'grid':
                dep_data[ii] = dep_data[ii][rl]

        data = self.var.func(*dep_data, **self.coords[rl], **self.kwargs)

        if self.var.save:
            with HDF5(f'{self.var.sim.pp_hdf5_path}/{key}.h5', 'a') as hdf5:
                hdf5.create_dataset(dset_path, data=data)

        if self.mem_load:
            self.mem_data[rl] = data

        return data


class TimeSeries():
    """Documentation for TimeSeries """

    var: "Variable"
    its: 'NDArray[np.int_]'
    data: 'NDArray[np.float_]'
    times: 'NDArray[np.float_]'
    restarts: 'NDArray[np.int_]'

    def __init__(self,
                 var: "Variable",
                 its: Optional['NDArray[np.int_]'] = None,
                 **kwargs):
        self.var = var
        self.its, self.times, self.data, self.restarts = self.var.sim.data_handler.get_time_series(
            self.var.key)
        self.its = self.its.astype(int)
        self.restarts = self.restarts.astype(int)
        if its is not None:
            self.its, inds, _ = np.intersect1d(
                self.its, its, return_indices=True)
            self.times = self.times[inds]
            self.data = self.data[inds]
            self.restarts = self.restarts[inds]

    @property
    def scaled_data(self) -> "'NDArray[np.float_]'":
        return self.data * self.var.scale_factor

    def __len__(self):
        return len(self.its)

    def __repr__(self):
        return (f"TimeSeries: {self.var.key};"
                f"time={self.times[0]*Units['Time']:.2f} - {self.times[-1]*Units['Time']:.2f}ms,"
                f"length={len(self)}")


class PPTimeSeries(TimeSeries):
    """Documentation for PPTimeSeries"""
    kwargs: Dict[str, Any]

    def __init__(self,
                 var: "PPTimeSeriesVariable",
                 its: 'NDArray[np.int_]',
                 **kwargs):
        self.var = var
        self.kwargs = kwargs

        # checked for saved version
        if self.var.save:
            key = self.var.key
            for kk, item in self.kwargs.items():
                if kk in self.var.PPkeys:
                    key += f":{kk}={item}"
            if os.path.isfile(f'{self.var.sim.pp_hdf5_path}/time_series.h5'):
                with HDF5(f'{self.var.sim.pp_hdf5_path}/time_series.h5', 'r') as hdf5:
                    if key in hdf5:
                        if 'its' not in hdf5[key].attrs:
                            print(self.var.sim)
                        old_its = np.intersect1d(its, hdf5[key].attrs['its'])
                        new_its = np.setdiff1d(its, old_its)
                        if len(new_its) == 0:
                            self.its = its
                            inds = np.searchsorted(hdf5[key].attrs['its'], its)
                            self.data = hdf5[key][inds]
                            self.times = hdf5[key].attrs['times'][inds]
                            self.restarts = hdf5[key].attrs['restarts'][inds]
                            return
                        its = new_its

        if self.var.reduction is not None:
            data = self.var.reduction(dependencies=self.var.dependencies,
                                      func=self.var.func,
                                      its=its,
                                      var=self.var,
                                      **kwargs)
            if all(dvar.vtype == 'time' for dvar in self.var.dependencies):
                times = (dat := self.var.dependencies[0].get_data(
                    it=its, **kwargs)).times
                restarts = dat.restarts
            else:
                times = self.var.sim.get_time(its)
                restarts = self.var.sim.get_restart(its)

        else:
            dep_data = [dep.get_data(it=its, **self.kwargs)
                        for dep in self.var.dependencies]

            times = dep_data[0].times
            restarts = dep_data[0].restarts
            data = self.var.func(
                *[dd.data for dd in dep_data], times, **self.kwargs)

        with HDF5(f'{self.var.sim.pp_hdf5_path}/time_series.h5', 'a') as hdf5:
            if self.var.save and key in hdf5:
                self.its = np.sort(np.concatenate((its, old_its)))
                self.data = np.zeros_like(self.its, dtype=float)
                self.times = np.zeros_like(self.its, dtype=float)
                self.restarts = np.zeros_like(self.its, dtype=int)

                n_new = np.searchsorted(self.its, its)
                self.data[n_new] = data
                self.times[n_new] = times
                self.restarts[n_new] = restarts

                n_old = np.searchsorted(self.its, old_its)
                n_hdf5 = np.searchsorted(hdf5[key].attrs['its'], old_its)
                self.data[n_old] = hdf5[key][n_hdf5]
                self.times[n_old] = hdf5[key].attrs['times'][n_hdf5]
                self.restarts[n_old] = hdf5[key].attrs['restarts'][n_hdf5]

                del hdf5[key]

            else:
                self.data = data
                self.its = its
                self.times = times
                self.restarts = restarts

            if self.var.save:
                dset = hdf5.create_dataset(key, data=self.data)
                dset.attrs['its'] = self.its
                dset.attrs['times'] = self.times
                dset.attrs['restarts'] = self.restarts


class GWData(TimeSeries):
    """Documentation for GWData"""
    kwargs: Dict[str, Any]

    def __init__(self,
                 var: "PPTimeSeriesVariable",
                 its: 'NDArray[np.int_]',
                 **kwargs):
        self.var = var
        self.kwargs = kwargs

        self.times, self.data = self.var.func(self.var, **kwargs)
        self.its = var.sim.get_it(time=self.times)
        self.restarts = var.sim.get_restart(it=self.its)
        if np.any(np.diff(self.its) == 0):
            # if iereations are the same everything goes to shit
            self.its = np.arange(len(self.times))
            self.restarts = np.arange(len(self.times))
