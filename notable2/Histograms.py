from functools import reduce
from typing import TYPE_CHECKING, Optional
from tqdm import tqdm
from numpy.typing import NDArray
import numpy as np
import alpyne.uniform_interpolation as ui  # type: ignore

from .Utils import IterationError
from .RCParams import rcParams

if TYPE_CHECKING:
    from .Utils import RLArgument, Simulation


def getHist(sim: "Simulation",
            keys: list[str],
            it: Optional[int] = None,
            time: Optional[float] = None,
            rls: "RLArgument" = None,
            **kwargs):

    for key in ['dens', 'reduce-weights']:
        if key not in keys:
            keys = keys + [key]
    vars = {key: sim.get_variable(key) for key in keys}
    region = 'xz' if sim.is_cartoon else 'xyz'
    its = reduce(np.intersect1d, (var.available_its(region)
                 for var in vars.values()))
    if it is None and time is None:
        it = its[0]
    elif isinstance(it, (int, np.integer)):
        if it not in its:
            raise IterationError(
                f"Iteration {it} for Variables {vars} not in {sim}")
    elif isinstance(time, (int, float, np.number)):
        times = sim.get_time(its)
        if sim.t_merg is not None:
            times -= sim.t_merg
        it = its[times.searchsorted(time)]
    else:
        raise ValueError

    grid_funcs = {key: var.get_data(region=region, it=it, **kwargs)
                  for key, var in vars.items()}

    coords = grid_funcs['dens'].coords

    actual_rls = sim.expand_rl(rls, it)

    datas = {key: {rl: gf[rl] for rl in actual_rls}
             for key, gf in grid_funcs.items()}
    dats = {key: np.concatenate([dat.ravel() for dat in data.values()])
            for key, data in datas.items()}

    vols = []
    for rl in actual_rls:
        coord = coords[rl]
        dx = {ax: cc[1] - cc[0] for ax, cc in coord.items()}
        coord = dict(zip(coord, np.meshgrid(*coord.values(), indexing='ij')))
        if sim.is_cartoon:
            vol = 2*np.pi*dx['x']*dx['z']*np.abs(coord['x'])
        else:
            vol = dx['x']*dx['y']*dx['z']*np.ones_like(coord['x'])
        vols.append(vol.ravel())
    vols = np.concatenate(vols)

    mdat = vols*dats['dens']*dats['reduce-weights']

    return dats, mdat


def getEjectaHist(sim: "Simulation",
                  keys: list[str],
                  radius: float,
                  min_it: Optional[int] = None,
                  max_it: Optional[int] = None,
                  min_time: Optional[float] = None,
                  max_time: Optional[float] = None,
                  every: int = 1,
                  rls: "RLArgument" = None,
                  unbound: str = 'bernulli',
                  ):

    if (unbound is None) or (unbound == "bernulli"):
        bg = 'b'
        print('using Bernulli criterion')
    elif unbound == "geodesic":
        bg = ''
        print('using geodesic criterion')
    else:
        raise ValueError(f'Unbound criterion "{unbound}" not supported')

    if f'ej{bg}-flow' not in keys:
        keys.append(f'ej{bg}-flow')
    vars = {key: sim.get_variable(key) for key in keys}

    region = 'xz' if sim.is_cartoon else 'xyz'

    its = reduce(np.intersect1d, (var.available_its(region)
                 for var in vars.values()))
    if min_time is not None:
        min_it = vars[keys[0]].get_it(min_time, region=region, t_merg=True)
        its = its[(its >= min_it)]
    if max_time is not None:
        max_it = vars[keys[0]].get_it(max_time, region=region, t_merg=True)
        its = its[(its <= max_it)]
    its = its[::every]

    dth = np.pi/2/rcParams.surf_int_n_theta
    thetas = (np.arange(rcParams.surf_int_n_theta)+.5)*dth
    if sim.is_cartoon:
        phis = np.array([0])
    else:
        dph = np.pi*2/rcParams.surf_int_n_phi
        phis = (np.arange(rcParams.surf_int_n_phi) + .5)*dph

    thetas, phis = np.meshgrid(thetas, phis, indexing='ij')

    if sim.is_cartoon:
        sphere = {'x': radius * np.sin(thetas),
                  'z': radius * np.cos(thetas)}
    else:
        sphere = {
            'x': np.ravel(radius * np.sin(thetas)*np.cos(phis)),
            'y': np.ravel(radius * np.sin(thetas)*np.sin(phis)),
            'z': np.ravel(radius * np.cos(thetas))
        }

    coords = sim.get_coords(region=region, it=its[-1])
    for rl, coord in list(coords.items())[::-1]:
        if all(cc.max() > radius for cc in coord.values()):
            break
    else:
        raise ValueError(f"Radius {radius} is larger then domain")

    origin = np.array([cc[0] for cc in coord.values()])
    i_dx = 1/np.array([cc[1] - cc[0] for cc in coord.values()])
    interp_data = ui.getInterpPoints3D(
        sphere['x'], sphere['y'], sphere['z'], origin, i_dx)

    result: dict[str, dict[int, NDArray[np.float_]]] = {kk: {} for kk in keys}
    for it in tqdm(its, disable=sim.verbose):
        data = np.array([var.get_data(region=region, it=it)[rl]
                         for var in vars.values()])

        interp_values = ui.applyPoints3D(*interp_data, data)
        for kk, res in zip(keys, interp_values):
            result[kk][it] = res
    return result
