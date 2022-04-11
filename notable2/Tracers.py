from typing import Optional, TYPE_CHECKING
from functools import reduce
from operator import lt, gt, eq, ge, le
import numpy as np
from tqdm import tqdm
from scipy.integrate import ode
from scipy.interpolate import interp1d
from alpyne.uniform_interpolation import linterp2D, chinterp2D

from .RCParams import rcParams
from .Tracer_refinement import GridRefine_multi, get_surf_flux_3D

if TYPE_CHECKING:
    from .Utils import Simulation, RLArgument

############################################
# 9 point 2D Gaus quadrature parameters
############################################

qq = np.array([-1, 0, 1])*np.sqrt(3/5)
ww = np.array([5, 8, 5])/9

qx, qz = map(np.ravel, np.meshgrid(qq, qq))
ww = np.tensordot(ww, ww, axes=0).ravel()
############################################


def _even_dist(nn, weights):
    nr = nn*weights
    ns = nr.astype(int)
    rest = nn - sum(ns)
    inext = np.argmax(nr - ns)
    while rest > 0:
        ns[inext] += 1
        rest -= 1
        inext = np.argmax(nr-ns)
    assert rest == 0, f'{rest}'
    return ns


class Tracers():
    def __init__(self, sim,
                 save_path,
                 seed_path,
                 max_step=203.025,
                 to_trace=('dens', 'temp', 'ye'),
                 verbose=True,
                 end_time=None,
                 chunksize=2e9,
                 terminate_var=None,
                 terminate_val=None,
                 t_int_order=1,
                 x_int_order=1):
        """

        """
        self.sim = sim
        self.sim.warn_backup = False
        self.seed_path = seed_path
        self.save_path = save_path
        self.to_trace = tuple(to_trace)
        self.verbose = verbose
        self.seed_path = seed_path
        self.max_step = max_step

        self.terminate_var = terminate_var
        self.terminate_val = terminate_val

        self.x_int_order = x_int_order
        if x_int_order == 1:
            self.interpolator = linterp2D
        elif x_int_order == 3:
            self.interpolator = chinterp2D
        else:
            raise ValueError(f"Tnterpolation in space with order {x_int_order} not supported")

        if t_int_order == 1:
            self.off = 1
            self.t_int_kind = 'linear'
        elif t_int_order == 3:
            self.off = 2
            self.t_int_kind = 'cubic'
        else:
            raise ValueError(f"Time interpolation with order {t_int_order} not supported")

        self.init_grid()
        times, _ = self.sim.get_data_multi('Vx', 2, progress_bar=self.verbose)
        self.dt = max(np.diff(times))

        self.end_time = times[-self.off]-1e-2 if end_time is None else end_time

        if self.end_time >= times[-self.off]:
            raise ValueError("given end_time is to large")
        if self.end_time < times[self.off]:
            raise ValueError("given end_time is to small")

        self.init_time = self.init_tracers()

        i_init = np.searchsorted(times, min(self.init_time, self.end_time), side='right') - 1
        i_end = np.searchsorted(times, max(self.init_time, self.end_time), side='right') - 1
        n_its = i_end - i_init + 2*self.off

        size_per_it = sum(np.prod(self.shapes, axis=1))*8*(3+len(self.to_trace))
        its_per_step = int(chunksize/size_per_it) - 2*self.off
        self.n_t_steps = int(np.ceil(n_its/its_per_step))

        self.time_step = (self.end_time - self.init_time)/self.n_t_steps

        if self.verbose:
            print(f'integrating in {self.n_t_steps} time steps')

    def init_grid(self,):
        tmp = self.sim.get_data('rho', 2)

        self.n_rl = len(tmp.reflevels)

        self.origin = np.zeros((self.n_rl, 2), dtype=float)
        self.idx = np.zeros((self.n_rl, 2), dtype=float)
        self.shapes = np.zeros((self.n_rl, 2), dtype=int)
        self.rl_bounds = np.zeros((self.n_rl, 4), dtype=float)

        for (xx, zz), dd, rl in tmp:
            self.shapes[rl] = np.array(dd.shape)
            self.origin[rl] = np.array([xx[0], zz[0]])
            self.idx[rl] = xx[1]-xx[0]
            self.rl_bounds[rl] = np.array([xx[self.x_int_order-1], xx[-self.x_int_order],
                                           zz[self.x_int_order-1], zz[-self.x_int_order]])
        self.idx = self.idx**-1

        del tmp

    def init_data(self, min_time=None, max_time=None):
        keys = self.to_trace + ('Vx', 'Vy', 'Vz')
        self.data = {kk: {} for kk in keys}

        for kk in keys:
            self.times, data = self.sim.get_data_multi(kk, 2,
                                                       progress_bar=self.verbose,
                                                       min_time=min_time,
                                                       max_time=max_time)
            n_it = len(self.times)
            for rl in range(self.n_rl):
                shape = [n_it, *self.shapes[rl]]
                self.data[kk][rl] = np.zeros(shape, dtype=float)

            for ii, dat in enumerate(data):
                for rl in range(self.n_rl):
                    self.data[kk][rl][ii] = dat[rl][1]

    def init_tracers(self):

        self.tracers = []

        if self.terminate_val is not None:
            if self.terminate_var[-1] == '<':
                termop = lt
                termvar = self.terminate_var[: -1]
                if self.verbose:
                    print(f"Terminating at {termvar} < {self.terminate_val}")
            elif self.terminate_var[-1] == '>':
                termop = gt
                termvar = self.terminate_var[: -1]
                if self.verbose:
                    print(f"Terminating at {termvar} > {self.terminate_val}")
            elif self.terminate_var[-1] == '=':
                termop = eq
                termvar = self.terminate_var[: -1]
                if self.verbose:
                    print(f"Terminating at {termvar} = {self.terminate_val}")
            elif self.terminate_var[-2] == '>=':
                termop = ge
                termvar = self.terminate_var[: -2]
                if self.verbose:
                    print(f"Terminating at {termvar} >= {self.terminate_val}")
            elif self.terminate_var[-2] == '<=':
                termop = le
                termvar = self.terminate_var[: -2]
                if self.verbose:
                    print(f"Terminating at {termvar} <= {self.terminate_val}")
            else:
                raise ValueError

        seeds = np.loadtxt(self.seed_path)

        for num, xx, zz, t_init, weight in seeds:
            self.tracers.append(tracer(num=int(num),
                                       path=self.save_path,
                                       position=np.array([xx, 0, zz]),
                                       t_init=t_init,
                                       t_end=self.end_time,
                                       to_trace=self.to_trace,
                                       weight=weight,
                                       termvar=termvar,
                                       termop=termop,
                                       termval=self.terminate_val,
                                       data_getter=self.get_data,
                                       max_step=self.max_step))
        return seeds.T[3].max()

    def get_data(self, tt, pos, keys):
        ind = np.searchsorted(self.times, tt, side='right')

        for rl in range(self.n_rl-1, -1, -1):
            if (self.rl_bounds[rl][0] < pos[0] <= self.rl_bounds[rl][1] and
                    self.rl_bounds[rl][2] < pos[2] <= self.rl_bounds[rl][3]):
                break
        else:
            raise ValueError("Point is not in domain:\n"
                             f'{self.rl_bounds[rl][0]} < {pos[0]} <= {self.rl_bounds[rl][1]}\n'
                             f'{self.rl_bounds[rl][2]} < {pos[2]} <= {self.rl_bounds[rl][3]}')
        points = []
        for kk in keys:
            for ii in range(ind-self.off, ind+self.off):
                try:
                    points.append(self.data[kk][rl][ii])
                except IndexError as e:
                    print(ind-self.off, ind+self.off)
                    print(self.data[kk][rl].shape)
                    print(self.times.min())
                    print(self.times.max())
                    raise e

        data = np.array(points)

        res = self.interpolator(np.array([pos[0], ]),
                                np.array([pos[2], ]),
                                self.origin[rl],
                                self.idx[rl],
                                data).ravel()
        result = {kk: interp1d(self.times[ind-self.off: ind+self.off],
                               res[2*ii*self.off: 2*(ii+1)*self.off],
                               kind=self.t_int_kind)(tt) for ii, kk in enumerate(keys)}

        return result

    def integrate_all(self):
        for ii in range(self.n_t_steps):
            t_start = self.init_time + ii*self.time_step
            t_end = self.init_time + (ii+1)*self.time_step

            self.init_data(min(t_end, t_start)-self.off*self.dt,
                           max(t_end, t_start)+self.off*self.dt)

            if self.time_step > 0:
                active_tracers = [tr for tr in self.tracers
                                  if t_end > tr.t and not tr.done]
            else:
                active_tracers = [tr for tr in self.tracers
                                  if t_end < tr.t and not tr.done]

            for tr in tqdm(active_tracers,
                           desc=f'Integrating {ii+1}/{self.n_t_steps}: {t_start*.00493:4.2f}-{t_end*.00493:4.2f}ms',
                           unit='tracer',
                           ncols=100,
                           total=len(active_tracers),
                           leave=True,
                           disable=not self.verbose,):
                tr.integrate(t_end)

        success = [tr.successful() for tr in self.tracers]
        if not all(success):
            print(f'{len(self.tracers) - sum(success)} tracers failed')
        print(f'{sum([tr.done for tr in self.tracers])} tracers reached the termination criterion')

        for tr in self.tracers:
            tr.finalize()
        if self.verbose:
            print("Done")


class tracer(ode):

    def __init__(self,
                 num,
                 path,
                 position,
                 t_init,
                 t_end,
                 to_trace,
                 weight,
                 data_getter,
                 max_step=203.025,
                 termvar=None,
                 termval=None,
                 termop=None
                 ):

        self.num = num
        self.path = path
        self.t_end = t_end
        self.weight = weight
        self.get_data = data_getter

        self.termvar = termvar
        self.termval = termval
        self.termop = termop

        super().__init__(self.get_rhs)
        self.set_integrator('dopri5', nsteps=10000, max_step=max_step)
        self.set_initial_value(position, t=t_init)

        self.done = False
        self.xx = []
        self.phi = []
        self.zz = []
        self.time = []
        self.trace = {kk: [] for kk in to_trace}

        if self.termop is not None:
            self.set_solout(self.set_trace_terminate)
        else:
            self.set_solout(self.set_trace)

    def get_rhs(self, tt, pos):
        if np.isnan(pos[0]):
            raise RuntimeError(f"x is nan at t={tt} for tracer {self.num}")
        if np.isnan(pos[1]):
            raise RuntimeError(f"phi is nan at t={tt} for tracer {self.num}")
        if np.isnan(pos[2]):
            raise RuntimeError(f"z is nan at t={tt} for tracer {self.num}")

        dic = self.get_data(tt, pos, ['Vx', 'Vy', 'Vz'])
        return np.array([dic['Vx'], dic['Vy']/pos[0], dic['Vz']])

    def set_trace(self, tt, pos):
        self.xx.append(pos[0])
        self.phi.append(pos[1])
        self.zz.append(pos[2])
        self.time.append(tt)

        data = self.get_data(tt, pos, self.trace.keys())
        for kk in self.trace.keys():
            self.trace[kk].append(data[kk])

    def set_trace_terminate(self, tt, pos):
        self.xx.append(pos[0])
        self.phi.append(pos[1])
        self.zz.append(pos[2])
        self.time.append(tt)

        data = self.get_data(tt, pos, self.trace.keys())

        for kk in self.trace.keys():
            self.trace[kk].append(data[kk])

        if self.termop(self.trace[self.termvar][-1], self.termval):
            self.done = True
            return -1

    def finalize(self):

        self.time, isort = np.unique(self.time, return_index=True)
        self.xx = np.array(self.xx)[isort]
        self.phi = np.array(self.phi)[isort]
        self.zz = np.array(self.zz)[isort]
        for kk in self.trace.keys():
            self.trace[kk] = np.array(self.trace[kk])[isort]

        arr = [self.time, self.xx, self.phi, self.zz]
        arr += [self.trace[kk] for kk in self.trace.keys()]

        arr = np.array(arr)

        header = f'weight:{self.weight}\n'
        header += '0:t, 1:x, 2:phi, 3:z, '
        for ii, kk in enumerate(self.trace.keys()):
            header += f'{ii+4}:{kk}, '

        np.savetxt(f'{self.path}/tracer_{self.num:05d}.dat',
                   arr.T,
                   fmt='%16.8e',
                   newline='\n',
                   header=header)


def load_tracer(file_path):
    with open(file_path, 'r') as ff:
        header = ff.readline()
        header = header.replace('# ', '')
        mass = float(header.split(':')[1])
        header = ff.readline()
    header = header.replace('# ', '')
    header = header.split(',')
    keys = []
    for hh in header[: -1]:
        keys.append(hh.split(":")[1].strip())

    data = dict(zip(keys, np.loadtxt(file_path, unpack=True)))
    data['mass'] = mass
    return data


class DensitySeeds():

    def __init__(self,
                 sim,
                 n_tracer,
                 max_dx,
                 path,
                 init_it=None,
                 m_thresh_init=None,
                 conditions=None,
                 unbound=None,
                 max_dens=None,
                 n_files=1):
        self.sim = sim
        self.init_it = sim.its[2] if init_it is None else init_it
        t_init = sim.complete_time_it(None, self.init_it)[0]
        self.dens = sim.get_data('dens', 2, it=init_it, exclude_ghosts=False)
        if unbound in ['geodesic']:
            self.u_t = sim.get_data('u_t', 2, it=init_it, exclude_ghosts=False)
        if unbound in ['bernulli']:
            self.u_t = sim.get_data('hu_t', 2, it=init_it, exclude_ghosts=False)

        self.conditions = conditions if conditions is not None else []
        self.unbound = unbound
        self.max_dens = max_dens

        self.n_tracer = n_tracer
        self.max_dx = max_dx

        self.n_files = n_files
        self.path = path

        n_tr = self.init_grid()

        self.m_thresh = 1/self.n_tracer if m_thresh_init is None else m_thresh_init
        max_ref = 100
        print(f"{n_tr} tracers in initial grid with spacing {self.max_dx}")

        # Coarse refinement
        for _ in range(max_ref):
            n_tr = self.get_refined_grid()[0]

            if n_tr > self.n_tracer:
                break

            self.m_thresh /= max(self.n_tracer/n_tr, 2)
        else:
            print(f"Maximum refinemet levels reached with {n_tr} tracer")

        # Fine refinement
        step = self.m_thresh / 5
        for _ in range(max_ref):
            self.m_thresh += step
            n_tr, (xs, zs, ms) = self.get_refined_grid()

            if n_tr < self.n_tracer:
                self.m_thresh -= step
                step /= np.pi
            elif n_tr <= self.n_tracer*1.01:
                break
        else:
            print(f"Maximum refinemet levels reached with {n_tr} tracer")

        print()
        self.n_tr = n_tr

        ms *= 2*2*np.pi*xs

        print(f"Total tracer mass: {np.sum(ms):.5f}M")

        seeds = np.stack((np.arange(self.n_tr), xs[ms > 0], zs[ms > 0], t_init *
                          np.ones(sum(ms > 0), dtype=float), ms[ms > 0])).T
        self.output(seeds)

    def init_grid(self,):
        if self.unbound in ['geodesic', 'bernulli']:
            (xx, zz), _ = self.u_t[0]
        else:
            (xx, zz), _ = self.dens[0]

        xm, zm = np.meshgrid(xx, zz, indexing='ij', copy=False)

        mask = np.ones_like(xm).astype(bool)
        for cond in self.conditions:
            mask = mask & cond(xm, zm)

        xm = xm[mask]
        zm = zm[mask]

        mask = np.ones_like(xm).astype(bool)
        if self.unbound in ['bernulli', 'geodesic']:
            ut = self.u_t(xm, zm)
            mask = mask & (ut <= -1)
        if self.max_dens is not None:
            dd = self.dens(xm, zm)
            mask = mask & (dd <= self.max_dens)

        xm = xm[mask]
        zm = zm[mask]

        n_grid = max(xm.max()/self.max_dx, zm.max()/self.max_dx)

        xx = self.max_dx*(np.arange(n_grid)+.5)
        zz = self.max_dx*(np.arange(n_grid)+.5)

        xm, zm = np.meshgrid(xx, zz, indexing='ij', copy=False)
        dx = self.max_dx*np.ones_like(xm)
        dz = self.max_dx*np.ones_like(zm)

        self.ini_grid = tuple(map(np.ravel, (xm, zm, dx, dz)))
        return len(xm.ravel())

    def get_cell_mass(self, xm, zm, dx, dz):

        pos_x = np.concatenate([qx*d/2 + xc for xc, d in zip(xm, dx)])
        pos_z = np.concatenate([qz*d/2 + zc for zc, d in zip(zm, dz)])

        dds = self.dens(pos_x, pos_z).reshape(len(xm), len(ww))

        dm = np.tensordot(ww, dds, axes=(0, 1))*dx*dz/4

        return dm

    def get_refined_grid(self):
        x_fin, z_fin, m_fin = [], [], []

        x_old, z_old, dx_old, dz_old = [data.copy() for data in self.ini_grid]
        ref = True
        n_tr = 0
        ii = 0
        while np.any(ref):
            ii += 1
            m_old = self.get_cell_mass(x_old, z_old, dx_old, dz_old)

            ref = m_old >= self.m_thresh

            if ii % 2 == 1:
                dx_new = dx_old[ref]/2
                dz_new = dz_old[ref]
                x_new = np.concatenate([x_old[ref] + dx_new/2, x_old[ref] - dx_new/2])
                z_new = np.concatenate([z_old[ref], z_old[ref]])
            else:
                dx_new = dx_old[ref]
                dz_new = dz_old[ref]/2
                x_new = np.concatenate([x_old[ref], x_old[ref]])
                z_new = np.concatenate([z_old[ref] + dz_new/2, z_old[ref] - dz_new/2])

            dx_new = np.repeat(dx_new, 2)
            dz_new = np.repeat(dz_new, 2)

            x_old = x_old[~ref]
            z_old = z_old[~ref]
            dx_old = dx_old[~ref]
            dz_old = dz_old[~ref]
            m_old = m_old[~ref]
            assert np.all(m_old < self.m_thresh)

            mask = np.ones_like(x_old).astype(bool)
            for cond in self.conditions:
                mask = mask & cond(x_old, z_old)
            x_fin.append(x_old[mask])
            z_fin.append(z_old[mask])
            m_fin.append(m_old[mask])

            mask = np.ones_like(x_new).astype(bool)
            for cond in self.conditions:
                ccs = (cond(x_new-dx_new/2, z_new-dz_new/2),
                       cond(x_new+dx_new/2, z_new-dz_new/2),
                       cond(x_new-dx_new/2, z_new+dz_new/2),
                       cond(x_new+dx_new/2, z_new+dz_new/2),)
                mask = mask & reduce(np.logical_or, ccs)
            if self.unbound in ['geodesic', 'bernulli']:
                uts = (self.u_t(x_new, z_new),
                       self.u_t(x_new-dx_new/2, z_new-dz_new/2),
                       self.u_t(x_new+dx_new/2, z_new-dz_new/2),
                       self.u_t(x_new-dx_new/2, z_new+dz_new/2),
                       self.u_t(x_new+dx_new/2, z_new+dz_new/2))
                mask = mask & reduce(np.logical_or, (ut <= -1 for ut in uts))
            if self.max_dens is not None:
                dds = (self.dens(x_new, z_new),
                       self.dens(x_new-dx_new/2, z_new-dz_new/2),
                       self.dens(x_new+dx_new/2, z_new-dz_new/2),
                       self.dens(x_new-dx_new/2, z_new+dz_new/2),
                       self.dens(x_new+dx_new/2, z_new+dz_new/2))
                mask = mask & reduce(np.logical_or, (dd <= self.max_dens for dd in dds))

            x_new = x_new[mask]
            z_new = z_new[mask]
            dx_new = dx_new[mask]
            dz_new = dz_new[mask]
            x_old = x_new
            z_old = z_new
            dx_old = dx_new
            dz_old = dz_new

            n_tr = sum(map(len, x_fin))
            print(f"M_max = {self.m_thresh:.3e}: {n_tr} Tracer after {ii} refinment steps. "
                  f"{len(x_new)} tracers still open", end=20*' ' + '\r')

            if n_tr > self.n_tracer*1.01:
                break
            if len(x_old) == 0:
                break

        grid = tuple(map(np.concatenate, (x_fin, z_fin, m_fin)))
        print()

        return n_tr, grid

    def output(self, seeds):
        np.random.shuffle(seeds)

        # output tracers in n_files files
        fmt = "%07d "+4*"%.7e "
        header = (f"simulation={self.sim.basepath}\n"
                  "n     x             z             t_init         mass")

        nn = int(np.ceil(self.n_tr/self.n_files))

        for num in range(self.n_files):
            start = num*nn
            end = min(start+nn, self.n_tr)

            np.savetxt(f'{self.path}/seeds_{num:05d}.dat',
                       seeds[start: end],
                       fmt=fmt,
                       header=header)

        return seeds.T


def FluxSeeds(sim: "Simulation",
              n_tracers: int,
              radius: float,
              path: str,
              n_theta: Optional[int] = None,
              n_phi: Optional[int] = None,
              min_time: Optional[float] = None,
              max_time: Optional[float] = None,
              unbound: Optional[float] = None,
              n_files: int = 1):

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

    if (unbound is None) or (unbound == "bernulli"):
        bg = 'b'
        u_t = 'h-u_t'
        print('using Bernulli criterion')
    elif unbound == "geodesic":
        bg = ''
        u_t = 'u_t'
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

    # geometry
    if n_theta is None:
        n_theta = rcParams.surf_int_n_theta
    if n_phi is None:
        n_phi = rcParams.surf_int_n_phi

    print(f"aiming for {n_tracers:.2e} tracers")

    fluxes = [sim.get_data(f'ej{bg}-flow', it=it, region=region) for it in its]

    if sim.is_cartoon:
        raise ValueError("Cartoon tracers not yet implemented")
    else:
        gr = GridRefine_multi((n_phi, n_theta), fluxes, get_surf_flux_3D(radius, dt))
        grids, masses = gr.get_mthresh(mthresh_est, n_tracers, n_tracers//100)

    n_tr = [len(mm) for mm in masses]

    t_seeds = np.concatenate([tt*np.ones_like(mm) for tt, mm in zip(times, masses)])
    it_seeds = np.concatenate([it*np.ones_like(mm) for it, mm in zip(its, masses)])
    m_seeds = np.concatenate(masses)
    if sim.is_cartoon:
        thetas = np.concatenate([np.pi/2*grid for grid in grids])
        xx, zz = [radius * np.sin(thetas), radius * np.cos(thetas)]

        seeds = np.stack([xx, zz, it_seeds, t_seeds, m_seeds])
    else:
        phis, thetas = np.concatenate([np.pi*np.array([2, .5]) * grid for grid in grids]).T
        xx, yy, zz = [radius * np.sin(thetas)*np.cos(phis),
                      radius * np.sin(thetas)*np.sin(phis),
                      radius * np.cos(thetas)]

        seeds = np.stack([  # xx, yy
            phis, thetas, zz, it_seeds, t_seeds, m_seeds])

    print(f"ejected mass: {mtot:.4e}M")
    print(f"ejected tracer mass: {np.sum(seeds[-1]):.4e}M")

    return seeds

    # output tracers in n_files files
    if sim.is_cartoon:
        fmt = 2*"%.7e "+"%10d "+2*"%.7e"
        header = (f"simulation={sim.sim_path}\n"
                  "x             z             it_init       t_init         mass")
    else:
        fmt = 3*"%.7e "+"%10d "+2*"%.7e"
        header = (f"simulation={sim.sim_path}\n"
                  "x             y             z             it_init        t_init         mass")

    nn = int(np.ceil(n_tracers/n_files))

    for num in range(n_files):
        start = num*nn
        end = min(start+nn, n_tracers)

        np.savetxt(f'{path}/{sim.sim_name}_seeds_{num:05d}.dat',
                   seeds.T[start: end],
                   fmt=fmt,
                   header=header)

    print(f"Done")
    return seeds
