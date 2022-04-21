import numpy as np
from typing import TYPE_CHECKING, Union, Sequence, Callable, Optional
from scipy.integrate import simps  # type: ignore

from .RCParams import rcParams
from .Variable import TimeSeriesBaseVariable, GridFuncBaseVariable

if TYPE_CHECKING:
    from .Utils import (
        GridFuncVariable, PPGridFuncVariable, TimeSeriesVariable,
        PPTimeSeriesVariable, Simulation, PostProcVariable
    )
    from numpy.typing import NDArray


def integral(dependencies: Sequence[Union["GridFuncVariable",
                                          "PPGridFuncVariable",
                                          "TimeSeriesVariable",
                                          "PPTimeSeriesVariable"]],
             func: Callable,
             its: 'NDArray[np.int_]',
             var: "PostProcVariable",
             rls: Optional['NDArray[np.int_]'] = None,
             **kwargs) -> 'NDArray[np.float_]':

    region = 'xz' if var.sim.is_cartoon else 'xyz'

    result = np.zeros_like(its, dtype=float)

    for ii, it in enumerate(its):
        if var.sim.verbose:
            print(f"{var.sim.sim_name} - {var.key}: Integrating iteration {it} ({ii/len(its)*100:.1f}%)",
                  end=(20*' '+'\r' if var.sim.verbose == 1 else '\n'))

        weights = var.sim.get_data('reduce-weights', region=region, it=it)
        dep_data = []
        for dep in dependencies:
            if dep.vtype == 'grid':
                dep_data.append(dep.get_data(region=region, it=it, **kwargs))
            else:
                dep_data.append(dep.get_data(it=it, **kwargs))
        for rl in var.sim.expand_rl(rls, it=it):
            coord = dep_data[0].coords[rl]
            try:
                dx = {ax: abs(cc[1] - cc[0]) for ax, cc in coord.items()}
            except IndexError:
                dx = {ax: np.nan for ax in coord}
            coord = dict(zip(coord, np.meshgrid(*coord.values(), indexing='ij')))

            if var.sim.is_cartoon:
                vol = 2*np.pi*dx['x']*dx['z']*np.abs(coord['x'])
            else:
                vol = dx['x']*dx['y']*dx['z']

            integ = func(*[data if isinstance(data, (float, np.floating)) else data[rl]
                           for data in dep_data], **coord, **kwargs) * weights[rl]

            integ[~np.isfinite(integ)] = 0

            integ *= vol
            result[ii] += np.sum(integ)

    return result


def sphere_surface_integral(dependencies: Sequence[Union["GridFuncVariable", "PPGridFuncVariable"]],
                            func: Callable,
                            its: 'NDArray[np.int_]',
                            var: "PostProcVariable",
                            radius: float,
                            **kwargs) -> 'NDArray[np.float_]':

    region = 'xz' if var.sim.is_cartoon else 'xyz'

    dth = np.pi/2/rcParams.surf_int_n_theta
    thetas = (np.arange(rcParams.surf_int_n_theta)+.5)*dth
    if var.sim.is_cartoon:
        phis = np.array([0])
    else:
        dph = np.pi*2/rcParams.surf_int_n_phi
        phis = (np.arange(rcParams.surf_int_n_phi) + .5)*dph

    thetas, phis = np.meshgrid(thetas, phis, indexing='ij')

    if var.sim.is_cartoon:

        sphere = {'x': radius * np.sin(thetas),
                  'z': radius * np.cos(thetas)}
    else:
        sphere = {'x': radius * np.sin(thetas)*np.cos(phis),
                  'y': radius * np.sin(thetas)*np.sin(phis),
                  'z': radius * np.cos(thetas)}

    result = np.zeros_like(its, dtype=float)
    for ii, it in enumerate(its):
        if var.sim.verbose:
            print(f"{var.sim.sim_name} - {var.key}: Integrating iteration {it} ({ii/len(its)*100:.1f}%)",
                  end=(20*' '+'\r' if var.sim.verbose == 1 else '\n'))
        dep_data = [dep.get_data(region=region, it=it, **kwargs)
                    for dep in dependencies]
        int_vals = [dat(**sphere) for dat in dep_data]

        if var.sim.is_cartoon:
            vol = 2*np.pi * dth * radius**2 * np.sin(thetas)
        else:
            vol = dth * dph * radius**2 * np.sin(thetas)
        integ = func(*int_vals, **sphere, **kwargs)
        result[ii] += np.sum(integ[(mask := np.isfinite(integ))] * vol[mask])
    return result


def minimum(dependencies: Sequence[Union["GridFuncVariable",
                                         "PPGridFuncVariable"]],
            its: 'NDArray[np.int_]',
            var: "PostProcVariable",
            rls: Optional['NDArray[np.int_]'] = None,
            **kwargs) -> 'NDArray[np.float_]':

    region = 'xz' if var.sim.is_cartoon else 'xyz'

    if (len(dependencies) > 1) or (dependencies[0].vtype != 'grid'):
        raise ValueError

    result = np.zeros_like(its, dtype=float)

    for ii, it in enumerate(its):
        acutal_rls = var.sim.expand_rl(rls, it=it)
        if var.sim.verbose:
            print(f"{var.sim.sim_name} - {var.key}: getting min at iteration {it} ({ii/len(its)*100:.1f}%)",
                  end=(20*' '+'\r' if var.sim.verbose == 1 else '\n'))

        weights = var.sim.get_data('reduce-weights', region=region, it=it)
        dep = dependencies[0].get_data(region=region, it=it)

        tmp = []
        for rl in acutal_rls:
            dat = dep[rl]
            mask = (weights[rl] == 1.) & np.isfinite(dat)
            tmp.append(np.min(dat[mask]))
        result[ii] = min(tmp)
    return result


def maximum(dependencies: Sequence[Union["GridFuncVariable",
                                         "PPGridFuncVariable"]],
            its: 'NDArray[np.int_]',
            var: "PostProcVariable",
            rls: Optional['NDArray[np.int_]'] = None,
            **kwargs) -> 'NDArray[np.float_]':

    region = 'xz' if var.sim.is_cartoon else 'xyz'

    if (len(dependencies) > 1) or (dependencies[0].vtype != 'grid'):
        raise ValueError

    result = np.zeros_like(its, dtype=float)*np.nan

    for ii, it in enumerate(its):
        if var.sim.verbose:
            print(f"{var.sim.sim_name} - {var.key}: getting max at iteration {it} ({ii/len(its)*100:.1f}%)",
                  end=(20*' '+'\r' if var.sim.verbose == 1 else '\n'))

        weights = var.sim.get_data('reduce-weights', region=region, it=it)
        dep = dependencies[0].get_data(region=region, it=it)

        tmp = []
        for rl in var.sim.expand_rl(rls, it=it):
            dat = dep[rl]
            if weights[rl].shape != dat.shape:
                continue  # no idea why this happens sometimes
            mask = (weights[rl] == 1.) & np.isfinite(dat)
            if np.any(mask):
                tmp.append(np.max(dat[mask]))
        if len(tmp) > 0:
            result[ii] = max(tmp)
    return result


def mean(dependencies: Sequence[Union["GridFuncVariable",
                                      "PPGridFuncVariable"]],
         its: 'NDArray[np.int_]',
         var: "PostProcVariable",
         func: Callable,
         rls: Optional['NDArray[np.int_]'] = None,
         **kwargs) -> 'NDArray[np.float_]':

    region = 'xz' if var.sim.is_cartoon else 'xyz'

    result = np.zeros_like(its, dtype=float)

    ts_data = {dep.key: dep.get_data(it=its).data
               for dep in dependencies if isinstance(dep, TimeSeriesBaseVariable)}

    for ii, it in enumerate(its):
        if var.sim.verbose:
            print(f"{var.sim.sim_name} - {var.key}: getting mean at iteration {it} ({(ii+1)/len(its)*100:.1f}%)",
                  end=(20*' '+'\r' if var.sim.verbose == 1 else '\n'))
        weights = var.sim.get_data('reduce-weights', region=region, it=it)
        dens = var.sim.get_data('dens', region=region, it=it)
        dep_data = [dep.get_data(region=region, it=it)
                    if isinstance(dep,  GridFuncBaseVariable)
                    else ts_data[dep.key][ii] for dep in dependencies]

        tot_mass = 0
        for rl in var.sim.expand_rl(rls, it=it):
            coord = dep_data[0].coords[rl]
            dx = {ax: cc[1] - cc[0] for ax, cc in coord.items()}
            coord = dict(zip(coord, np.meshgrid(*coord.values(), indexing='ij')))

            if var.sim.is_cartoon:
                vol = 2*np.pi*dx['x']*dx['z']*np.abs(coord['x'])
            else:
                vol = dx['x']*dx['y']*dx['z']

            mass = vol*dens[rl]*weights[rl]
            dat = func(*[data if isinstance(data, (float, np.floating)) else data[rl]
                         for data in dep_data], **coord, **kwargs) * mass

            mask = np.isfinite(dat) & (dat != 0.)
            result[ii] += np.sum(dat[mask])
            tot_mass += np.sum(mass[mask])
        result[ii] /= tot_mass
    return result
