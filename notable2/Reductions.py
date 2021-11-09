import numpy as np
from typing import TYPE_CHECKING, Union, Sequence, Callable, Optional
from numpy.typing import NDArray
from scipy.integrate import simps

from .RCParams import rcParams

if TYPE_CHECKING:
    from .Utils import GridFuncVariable, PPGridFuncVariable, TimeSeriesVariable, PPTimeSeriesVariable, Simulation


def integral(dependencies: Sequence[Union["GridFuncVariable",
                                          "PPGridFuncVariable",
                                          "TimeSeriesVariable",
                                          "PPTimeSeriesVariable"]],
             func: Callable,
             its: NDArray[np.int_],
             var: "PostProcVariable",
             rls: Optional[NDArray[np.int_]] = None,
             **kwargs) -> NDArray[np.float_]:

    region = 'xz' if var.sim.is_cartoon else 'xyz'
    if rls is None:
        rls = var.sim.rls

    result = np.zeros_like(its, dtype=float)

    for ii, it in enumerate(its):
        if var.sim.verbose:
            print(f"{var.sim.sim_name} - {var.key}: Integrating iteration {it} ({ii/len(its)*100:.1f}%)",
                  end=('\r' if var.sim.verbose == 1 else '\n'))
        weights = var.sim.get_data('reduce-weights', region=region, it=it)
        dep_data = []
        for dep in dependencies:
            if dep.vtype == 'grid':
                dep_data.append(dep.get_data(region=region, it=it, **kwargs))
            else:
                dep_data.append(dep.get_data(it=it, **kwargs))

        for rl in rls:
            coord = dep_data[0].coords[rl]
            dx = {ax: cc[1] - cc[0] for ax, cc in coord.items()}
            coord = dict(zip(coord, np.meshgrid(*coord.values(), indexing='ij')))

            if var.sim.is_cartoon:
                vol = 2*np.pi*dx['x']*dx['z']*np.abs(coord['x'])
            else:
                vol = dx['x']*dx['y']*dx['z']

            integ = [data if isinstance(data, (float, np.floating)) else data[rl]
                     for data in dep_data]

            integ = func(*integ, **coord, **kwargs) * weights[rl]

            integ[~np.isfinite(integ)] = 0

            integ *= vol
            result[ii] += np.sum(integ[np.isfinite(integ)])

    return result


def sphere_surface_integral(dependencies: Sequence[Union["GridFuncVariable", "PPGridFuncVariable"]],
                            func: Callable,
                            its: NDArray[np.int_],
                            var: "PostProcVariable",
                            radius: float,
                            **kwargs) -> NDArray[np.float_]:

    region = 'xz' if var.sim.is_cartoon else 'xyz'

    thetas = np.linspace(0, np.pi/2, rcParams.surf_int_n_theta)
    dth = thetas[1] - thetas[0]
    if var.sim.is_cartoon:
        phis = np.array([0])
    else:
        phis = np.linspace(0, np.pi/2, rcParams.surf_int_n_phi)
        dph = phis[1] - phis[0]

    thetas, phis = np.meshgrid(thetas, phis, indexing='ij')

    sphere = {'x': radius * np.cos(thetas)*np.cos(phis),
              'y': radius * np.cos(thetas)*np.sin(phis),
              'z': radius * np.sin(thetas)}

    relevant_rls: dict[int, list[int]] = {it: [] for it in its}
    for it in its:
        coords = var.sim.get_coords(region=region, it=it)
        for rl in sorted(coords, reverse=True):
            coord = coords[rl]
            if not any(cc.max() >= radius for cc in coord.values()):
                continue
            relevant_rls[it].append(rl)
            if all(cc.max()*np.sqrt(2) >= radius for cc in coord.values()):
                break
        else:
            raise RuntimeError(f"Radius {radius} not fully contained in domain of Var.Simulation {var.sim} at it {it}")

    result = np.zeros_like(its, dtype=float)

    for ii, (it, rls) in enumerate(relevant_rls.items()):
        if var.sim.verbose:
            print(f"{var.sim.sim_name} - {var.key}: Integrating iteration {it} ({ii/len(its)*100:.1f}%)",
                  end=('\r' if var.sim.verbose == 1 else '\n'))
        dep_data = [dep.get_data(region=region, it=it, **kwargs)
                    for dep in dependencies]
        for rl in rls:
            int_vals = [dat(**sphere) for dat in dep_data]

            if var.sim.is_cartoon:
                vol = 2*np.pi * dth * radius * sphere['z']
            else:
                vol = dth * dph * radius * sphere['z']
            integ = func(*int_vals, **sphere, **kwargs)
            result[ii] += np.sum(integ[(mask := np.isfinite(integ))] * vol[mask])
    return result
