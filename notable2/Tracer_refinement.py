from itertools import product
from typing import Callable, Optional
import numpy as np
from numpy.typing import NDArray
from matplotlib.colors import LogNorm
from scipy.optimize import bisect

from notable2.DataObjects import GridFunc

############################################
# 9 point 2D Gaus quadrature parameters
############################################

qq = np.array([-.5, 0, .5])*np.sqrt(3/5)
ww = np.array([5, 8, 5])/18

qx, qy = map(np.ndarray.flatten, np.meshgrid(qq, qq))
ww = np.tensordot(ww, ww, axes=0).ravel()


class GridRefine:

    def __init__(self,
                 grid_shape: tuple[int, ...],
                 density: GridFunc,
                 get_mass_func: Callable,
                 ):
        self.dens = density
        self.mass_getter = get_mass_func

        self.dim = len(grid_shape)

        dx = np.meshgrid(*(np.ones(nn)/nn for nn in grid_shape), indexing='ij')
        self.dx = np.stack([xx.flatten() for xx in dx], axis=-1)

        grid = np.meshgrid(*(np.linspace(1/2/nn, 1-1/2/nn, nn) for nn in grid_shape), indexing='ij')
        self.grid = np.stack([xx.flatten() for xx in grid], axis=-1)

        self.mass = self.mass_getter(self.grid, self.dx, self.dens)
        mask = np.isfinite(self.mass)
        self.grid = self.grid[mask]
        self.dx = self.dx[mask]
        self.mass = self.mass[mask]
        self.max_mass = np.max(self.mass)

    def refine_grid(self,
                    thresh_mass: float,
                    max_n_tracer: Optional[int] = None,
                    ) -> tuple[NDArray[np.float_], NDArray[np.float_]]:
        max_ref_steps = 6
        ref_step = 0
        ref_mask = self.mass > thresh_mass
        mass, grid, dx = self.mass, self.grid, self.dx
        n_tracer = len(mass)
        while np.any(ref_mask) and ref_step <= max_ref_steps:
            # print(f'steps: {ref_step}   ', end='\r')
            new_grid = [grid[~ref_mask]]
            new_dx = [dx[~ref_mask]]

            ref_grid = grid[ref_mask]
            ref_dx = dx[ref_mask]

            for dx in product(*[[-1/4, 1/4] for _ in range(self.dim)]):
                new_grid += [ref_grid + np.array(dx)*ref_dx]
                new_dx += [ref_dx/2]

            grid = np.concatenate(new_grid)
            dx = np.concatenate(new_dx)
            mass = self.mass_getter(grid, dx, self.dens)
            inside_mask = np.isfinite(mass)

            mass = mass[inside_mask]
            grid = grid[inside_mask]
            dx = dx[inside_mask]

            n_tracer = len(mass)

            ref_mask = mass > thresh_mass
            ref_step += 1
            if max_n_tracer is not None and n_tracer >= max_n_tracer:
                break
        # print(f'm_thresh = {thresh_mass:.2e} reached after {ref_step} steps. {len(mass)} points.')
        return grid, mass

    def get_mthresh(self, n_tracers, tol):
        if len(self.mass) > n_tracers-tol:
            return self.grid, self.mass

        class func:
            def __init__(self, refine_func):
                self.mass = []
                self.grid = []
                self.diffs = []
                self.refine_grid = refine_func

            def __call__(self, mm):
                mm = 10**mm
                if mm <= 1e-50:
                    return np.inf
                grid, mass = self.refine_grid(mm, n_tracers-tol)
                diff = len(mass) - n_tracers

                self.grid.append(grid)
                self.mass.append(mass)
                self.diffs.append(diff)

                if abs(diff) < tol:
                    return 0
                return diff

        ff = func(self.refine_grid)
        try:
            m_th = brentq(ff, -50, np.log10(self.mass.max()), maxiter=50, )
        except RuntimeError:
            pass
        ii = np.argmin(np.abs(ff.diffs))

        # print(f'aimed for {n_tracers} got {len(ff.mass[ii])}')
        return ff.grid[ii], ff.mass[ii]


class GridRefine_multi:

    def __init__(self,
                 grid_shape: tuple[int, ...],
                 densities: list[GridFunc],
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
                    ) -> tuple[list[NDArray[np.float_]], list[NDArray[np.float_]]]:
        max_ref_steps = 6
        ref_step = 0

        grids = []
        dxs = []
        masses = []
        for dens in self.dens:
            dx = np.stack(np.meshgrid(*(np.ones(nn)/nn for nn in self.grid_shape), indexing='ij'), axis=-1)
            grid = np.stack(np.meshgrid(*((np.arange(nn)+.5)/nn
                            for nn in self.grid_shape), indexing='ij'), axis=-1)
            mass = self.mass_getter(grid, dx, dens)

            mask = np.isfinite(mass)

            dxs.append(dx[mask])
            grids.append(grid[mask])
            masses.append(mass[mask])

        ref_mask = [mm > thresh_mass for mm in masses]
        n_tracer = sum(len(mm) for mm in masses)

        while any(np.any(rfm) for rfm in ref_mask) and ref_step <= max_ref_steps:
            # print(f'steps: {ref_step} currently {n_tracer} tracers')
            for ii, (grid, dx, mass, rfm, dens) in enumerate(zip(grids, dxs, masses, ref_mask, self.dens)):
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
                inside_mask = np.isfinite(mass)

                masses[ii] = mass[inside_mask]
                grids[ii] = grid[inside_mask]
                dxs[ii] = dx[inside_mask]

            ref_mask = [mm > thresh_mass for mm in masses]
            ref_step += 1
            n_tracer = sum(len(mm) for mm in masses)
            if max_n_tracer is not None and n_tracer >= max_n_tracer:
                break
        # print(f'm_thresh = {thresh_mass:.5e} reached after {ref_step} steps. {n_tracer:.3e} tracers.')
        return grids, masses

    def get_mthresh(self, mthres_estimate, n_tracers, tol):
        if self.ngrids*np.prod(self.grid_shape) > n_tracers-tol:
            return self.init_grid, self.init_mass

        class func:
            def __init__(self, refine_func):
                self.mass = []
                self.grid = []
                self.diffs = []
                self.step = 0
                self.refine_grid = refine_func

            def __call__(self, mm):
                if mm == 0.:
                    return np.inf
                self.step += 1
                grids, masses = self.refine_grid(mm, n_tracers-tol)
                n_cur = sum(len(mass) for mass in masses)
                diff = n_cur - n_tracers

                self.grid.append(grids)
                self.mass.append(masses)
                self.diffs.append(diff)

                print(f"{n_cur:.2e} tracers with m_thresh = {mm:.5e} ({self.step} steps)", end="\r")

                if abs(diff) < tol:
                    print()
                    return 0
                return diff

        ff = func(self.refine_grid)
        try:
            m_th = bisect(ff, 0, mthres_estimate*10, maxiter=50, xtol=1e-20)
        except RuntimeError:
            pass
        ii = np.argmin(np.abs(ff.diffs))

        # print(f'aimed for {n_tracers} got {len(ff.mass[ii])}')
        return ff.grid[ii], ff.mass[ii]


def get_surf_flux_3D(radius, dt):
    def surf(grid, dx, dens):
        xx, yy = grid.T
        dx, dy = dx.T

        dphi = dx*np.pi*2
        dtheta = dy*np.pi/2
        phis = xx*np.pi*2
        thetas = yy*np.pi/2

        area = np.sin(thetas)*radius**2*dtheta*dphi

        # thetas = np.stack([qx*dd + xc for xc, dd in zip(thetas, dtheta)])
        # phis = np.stack([qy*dd + yc for yc, dd in zip(phis, dphi)])

        # dtheta = np.stack([dtheta]*9, axis=1)
        # dphi = np.stack([dphi]*9, axis=1)

        points = {'x': radius * np.sin(thetas)*np.cos(phis),
                  'y': radius * np.sin(thetas)*np.sin(phis),
                  'z': radius * np.cos(thetas)}

        flux = dens(**points)
        # flux = np.matmul(flux, ww)

        return (flux*area*dt).T
    return surf
