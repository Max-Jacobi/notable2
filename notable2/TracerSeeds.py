from itertools import product
from typing import Callable, Optional, TYPE_CHECKING, List, Tuple
import numpy as np
from scipy.optimize import bisect


from notable2.DataObjects import GridFunc

if TYPE_CHECKING:
    from .Utils import Simulation

############################################
# 9 point 2D Gaus quadrature parameters
############################################

qq = np.array([-.5, 0, .5])*np.sqrt(3/5)
ww = np.array([5, 8, 5])/18

ww = np.array([0.5688888888888889, 0.4786286704993665,
               0.4786286704993665, 0.2369268850561891, 0.2369268850561891])/2
qq = np.array([0, -0.5384693101056831, 0.5384693101056831, -
               0.9061798459386640, 0.9061798459386640])/2


qx, qy, qz = map(np.ndarray.flatten, np.meshgrid(qq, qq, qq))
ww = np.tensordot(
    np.tensordot(ww, ww, axes=0),
    ww, axes=0
).ravel()


def densSeeds(sim: "Simulation",
              n_tracers: int,
              it: int,
              inner_r: float,
              outer_r: float,
              base_shape: tuple[int, int, int] = (10, 10, 5),
              unbound: Optional[str] = None):

    time = sim.get_time(it)
    region = 'xz' if sim.is_cartoon else 'xyz'

    if (unbound is None) or (unbound == "bernulli"):
        bg = 'b'
        print('using Bernulli criterion')
    elif unbound == "geodesic":
        bg = ''
        print('using geodesic criterion')
    else:
        raise ValueError(f'Unbound criterion "{unbound}" not supported')

    mtot = sim.get_data("M-ejb-in-radius", inner_r=inner_r,
                        outer_r=outer_r, it=it)
    mthresh_est = mtot/n_tracers*2

    print(f"aiming for {n_tracers:.2e} tracers")

    dens = sim.get_data(f'ej{bg}-dens', it=it,
                        region=region, inner_r=inner_r, outer_r=outer_r)

    if sim.is_cartoon:
        raise ValueError("Cartoon tracers not yet implemented")
    else:
        gr = GridRefine(base_shape,
                        [dens],
                        get_dens_3D(inner_r, outer_r),
                        )

    grids, dxs, masses = gr.get_mthresh(mthresh_est,
                                        n_tracers,
                                        n_tracers//500,)
    # steps, grids, dxs, masses = gr.refine_grid(mthresh_est)

    m_seeds = masses[0]
    grid = grids[0]
    dx = dxs[0]

    if sim.is_cartoon:
        coords = []
    else:
        grid = (np.random.rand(*dx.shape)-.5)*dx + grid
        xx = 2*outer_r*grid[..., 0] - outer_r
        yy = 2*outer_r*grid[..., 1] - outer_r
        zz = outer_r*grid[..., 2]

    rr = np.sqrt(xx**2 + yy**2 + zz**2)
    out_mask = rr > outer_r
    out_fac = rr[out_mask]/outer_r
    xx[out_mask] /= out_fac
    yy[out_mask] /= out_fac
    zz[out_mask] /= out_fac
    in_mask = rr < inner_r
    print(f"{np.sum(in_mask)} inside")
    in_fac = rr[in_mask]/inner_r
    xx[in_mask] /= in_fac
    yy[in_mask] /= in_fac
    zz[in_mask] /= in_fac

    num_seeds = np.arange(len(m_seeds))
    it_seeds = np.ones(len(m_seeds), dtype=int)*it
    t_seeds = np.ones(len(m_seeds))*time

    seeds = np.stack([num_seeds, xx, yy, zz, it_seeds, t_seeds, m_seeds])

    print(f"ejected mass: {mtot:.4e}M")
    print(f"ejected tracer mass: {np.sum(seeds[-1]):.4e}M")

    return seeds


def fluxSeeds(sim: "Simulation",
              n_tracers: int,
              radius: float,
              every: int = 1,
              n_theta: int = 5,
              n_phi: int = 20,
              min_time: Optional[float] = None,
              max_time: Optional[float] = None,
              unbound: Optional[str] = None):

    region = 'xz' if sim.is_cartoon else 'xyz'
    # find the relevant reflevels
    tmp_var = sim.get_variable('rho')
    its = tmp_var.available_its(region=region)
    if min_time is not None:
        min_it = tmp_var.get_it(min_time, region=region, t_merg=True)
        its = its[(its >= min_it)]
    if max_time is not None:
        max_it = tmp_var.get_it(max_time, region=region, t_merg=True)
        its = its[(its <= max_it)]

    its = its[::every]

    if (unbound is None) or (unbound == "bernulli"):
        bg = 'b'
        print('using Bernulli criterion')
    elif unbound == "geodesic":
        bg = ''
        print('using geodesic criterion')
    else:
        raise ValueError(f'Unbound criterion "{unbound}" not supported')

    # calculate the number of tracers per timestep
    mdot_data = sim.get_data(f'M-ej{bg}-esc-dot', radius=radius, it=its)
    times = mdot_data.times
    mdot = mdot_data.data

    dt = times[1] - times[0]
    mtot = dt*sum(mdot)

    mthresh_est = mtot/n_tracers*2

    print(f"aiming for {n_tracers:.2e} tracers")

    fluxes = [sim.get_data(f'ej{bg}-flow', it=it, region=region) for it in its]

    if sim.is_cartoon:
        raise ValueError("Cartoon tracers not yet implemented")
    else:
        gr = GridRefine((n_phi, n_theta), fluxes, get_surf_flux_3D(radius, dt))
        # grids, dxs, masses = gr.get_mthresh(mthresh_est, n_tracers, n_tracers//100)
        _, grids, dxs, masses = gr.refine_grid(mthresh_est, 2*n_tracers)

    it_seeds = np.concatenate([it*np.ones_like(mm)
                              for it, mm in zip(its, masses)])
    t_seeds = np.concatenate([tt*np.ones_like(mm)
                             for tt, mm in zip(times, masses)])
    m_seeds = np.concatenate(masses)

    isort = np.argsort(it_seeds)

    if sim.is_cartoon:
        thetas = np.concatenate([np.pi/2*grid for grid in grids])
        thetas = thetas[isort]
        coords = [radius * np.sin(thetas), radius * np.cos(thetas)]

    else:
        grids = [(np.random.rand(*dx.shape)-.5)*dx +
                 grid for grid, dx in zip(grids, dxs)]
        phis = np.concatenate([2*np.pi*grid[:, 0] for grid in grids])
        thetas = np.concatenate([np.arccos(grid[:, 1]) for grid in grids])

        phis = phis[isort]
        thetas = thetas[isort]

        coords = [radius * np.sin(thetas)*np.cos(phis),
                  radius * np.sin(thetas)*np.sin(phis),
                  radius * np.cos(thetas)]

    it_seeds = it_seeds[isort]
    t_seeds = t_seeds[isort]
    m_seeds = m_seeds[isort]
    num_seeds = np.arange(len(it_seeds))

    seeds = np.stack([num_seeds, *coords, it_seeds, t_seeds, m_seeds])

    print(f"ejected mass: {mtot:.4e}M")
    print(f"ejected tracer mass: {np.sum(seeds[-1]):.4e}M")

    return seeds


def saveSeeds(path, seeds, sim, n_files):
    n_tracers = len(seeds[0])

    # output tracers in n_files files
    if sim.is_cartoon:
        header = f"{'num':>6s}{'x':>15s}{'z':>15s}{'it':>10s}{'t':>15s}{'m':>15s}"
        fmt = "%8d"+2*"%15.7e"+"%10d"+2*"%15.7e"
    else:
        header = f"{'num':>6s}{'x':>15s}{'y':>15s}{'z':>15s}{'it':>10s}{'t':>15s}{'m':>15s}"
        fmt = "%8d"+3*"%15.7e"+"%10d"+2*"%15.7e"

    nn = int(np.ceil(n_tracers/n_files))

    for num in range(n_files):
        start = num*nn
        end = min(start+nn, n_tracers)

        np.savetxt(f'{path}/{sim.sim_name}_seeds_{num:05d}.dat',
                   seeds.T[start: end],
                   fmt=fmt,
                   header=header)


class GridRefine:

    def __init__(self,
                 grid_shape: Tuple[int, ...],
                 densities: List[GridFunc],
                 get_mass_func: Callable,
                 ):
        self.dens = densities
        self.mass_getter = get_mass_func
        self.grid_shape = grid_shape

        self.dim = len(grid_shape)
        self.ngrids = len(self.dens)

    def refine_grid(self,
                    thresh_mass: float,
                    max_n_tracer: Optional[int] = None,
                    ):
        max_ref_steps = 6
        ref_step = 0

        grids = []
        dxs = []
        masses = []
        for dens in self.dens:
            dx = np.stack(np.meshgrid(
                *(np.ones(nn)/nn for nn in self.grid_shape),
                indexing='ij'),
                axis=-1)
            grid = np.stack(np.meshgrid(
                *((np.arange(nn)+.5)/nn for nn in self.grid_shape),
                indexing='ij'),
                axis=-1)
            dx = np.array([dd.flatten() for dd in dx.T]).T
            grid = np.array([gg.flatten() for gg in grid.T]).T

            mass = self.mass_getter(grid, dx, dens)

            mask = np.isfinite(mass) & (mass > 0.)

            dxs.append(dx[mask])
            grids.append(grid[mask])
            masses.append(mass[mask])

        n_tracer = sum(len(mm) for mm in masses)

        ref_mask = [mm > thresh_mass for mm in masses]

        while (any(np.any(rfm) for rfm in ref_mask)
               and ref_step <= max_ref_steps):
            print(f"step: {ref_step}")
            for ii, (grid, dx, mass, rfm, dens) in enumerate(zip(grids,
                                                                 dxs,
                                                                 masses,
                                                                 ref_mask,
                                                                 self.dens)):

                if all(~rfm):
                    continue

                new_grid = [grid[~rfm]]
                new_dx = [dx[~rfm]]

                ref_grid = grid[rfm]
                ref_dx = dx[rfm]

                for dd in product(*[[-1/4, 1/4] for _ in range(self.dim)]):
                    new_grid += [ref_grid + np.array(dd)*ref_dx]
                    new_dx += [ref_dx/2]

                grid = np.concatenate(new_grid)
                dx = np.concatenate(new_dx)
                mass = self.mass_getter(grid, dx, dens)
                mask = np.isfinite(mass) & (mass > 0.)

                masses[ii] = mass[mask]
                grids[ii] = grid[mask]
                dxs[ii] = dx[mask]

            ref_mask = [mm > thresh_mass for mm in masses]
            nref = sum(sum(rfm) for rfm in ref_mask)

            ref_step += 1
            n_tracer = sum(len(mm) for mm in masses)
            if max_n_tracer is not None and n_tracer >= max_n_tracer:
                break
        return ref_step, grids, dxs, masses

    def get_mthresh(self, mthres_estimate, n_tracers, tol, **kwargs):
        class func:
            def __init__(self, refine_func):
                self.mass = []
                self.grid = []
                self.dx = []
                self.diffs = []
                self.refine_grid = refine_func

            def __call__(self, mm):
                if mm == 0.:
                    return np.inf
                steps, grids, dxs, masses = self.refine_grid(
                    mm, **kwargs)
                n_cur = sum(len(mass) for mass in masses)
                diff = n_cur - n_tracers

                self.grid.append(grids)
                self.dx.append(dxs)
                self.mass.append(masses)
                self.diffs.append(diff)

                print(
                    f"{n_cur:.2e} tracers with m_thresh = {mm:.5e}, {steps} refinement steps")

                if abs(diff) < tol:
                    print()
                    return 0
                return diff

        ff = func(self.refine_grid)
        try:
            bisect(ff, 0, mthres_estimate*3, maxiter=50, xtol=1e-20)
        except RuntimeError:
            pass
        ii = np.argmin(np.abs(ff.diffs))

        # print(f'aimed for {n_tracers} got {len(ff.mass[ii])}')
        return ff.grid[ii], ff.dx[ii], ff.mass[ii]


def get_dens_3D(inner_r, outer_r):
    def dens(grid, dx, rho):
        xx, yy, zz = grid[..., 0], grid[..., 1], grid[..., 2]
        dx, dy, dz = dx[..., 0], dx[..., 1], dx[..., 2]

        vol = dx*dy*dz * 4 * outer_r**3

        xx = np.array([pp + dd*qx for pp, dd in zip(xx, dx)])
        yy = np.array([pp + dd*qy for pp, dd in zip(yy, dy)])
        zz = np.array([pp + dd*qz for pp, dd in zip(zz, dz)])

        xx = xx*2*outer_r - outer_r
        yy = yy*2*outer_r - outer_r
        zz = zz*outer_r

        res = (rho(x=xx, y=yy, z=zz)*ww).sum(axis=-1)

        return (vol*res)
    return dens


def get_surf_flux_3D(radius, dt):
    def surf(grid, dx, dens):
        xx, yy = grid.T
        dx, dy = dx.T

        dphi = dx*np.pi*2
        dtheta = (1-yy**2)**-.5*dy
        phis = xx*np.pi*2
        thetas = np.arccos(yy)

        area = np.sin(thetas)*radius**2*dtheta*dphi

        points = {'x': radius * np.sin(thetas)*np.cos(phis),
                  'y': radius * np.sin(thetas)*np.sin(phis),
                  'z': radius * np.cos(thetas)}

        flux = dens(**points)
        return (flux*area*dt).T
    return surf
