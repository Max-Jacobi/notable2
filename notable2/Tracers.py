import re
from functools import reduce
import numpy as np
import scipy.integrate as sint
from scipy.interpolate import interp1d


class TracerBunch():
    def __init__(self,
                 sim,
                 save_path,
                 seed_path,
                 max_step=None,
                 to_trace=('rho', 'temp', 'ye'),
                 verbose=False,
                 chunksize=2e9,
                 terminate_var=None,
                 terminate_val=None,
                 t_int_order=1):
        """
        A bunch of tracers. Used to coordinate the calculation of multiple tracers in bunches based on memory allocation.
        """

        self.sim = sim
        self.seed_path = seed_path
        self.save_path = save_path
        self.to_trace = tuple(to_trace)
        self.verbose = verbose
        self.seed_path = seed_path
        self.chunksize = chunksize
        self.terminate_var = terminate_var
        self.terminate_val = terminate_val

        self.dats = {}
        self.region = 'xy' if self.sim.is_cartoon else 'xyz'

        if t_int_order == 1:
            self.off = 1
            self.t_int_kind = 'linear'
        elif t_int_order == 3:
            self.off = 2
            self.t_int_kind = 'cubic'
        else:
            raise ValueError(f"Time interpolation with order {t_int_order} not supported")

        self.vars = {var: self.sim.get_variable(var)
                     for var in self.to_trace+('V^x', 'V^y', 'V^z')}
        self.its = reduce(np.intersect1d,
                          (var.available_its(self.region)
                           for var in self.vars.values()))
        self.times = self.sim.get_time(self.its)
        self.max_step = self.times[1]-self.times[0] if max_step is None else max_step

        max_t = self.init_tracers()

        i_start = self.times.searchsorted(max_t)
        self.times = self.times[:i_start+self.off]
        self.its = self.its[:i_start+self.off]

        self.i_start = len(self.times)-1

    def load_chunk(self, i_start):
        large_it = self.its[i_start]
        cur_size = 0
        for it, dat in self.dats.items():
            if it > large_it:
                del dat
            else:
                for dd in dat.values():
                    cur_size += sum(md.size*md.itemsize for md in dd.mem_data.values())
        self.dats = {}
        ii = i_start
        for it in self.its[i_start::-1]:
            if it not in self.dats:
                self.dats[it] = {kk: var.get_data(region=self.region, it=it)
                                 for kk, var in self.vars.items()}
                for kk in self.dats[it]:
                    self.dats[it][kk].mem_load = True
                    cc = self.dats[it][kk].coords[0]
                    ss = len(cc['x'])*len(cc['y'])*len(cc['z'])*8
                    cur_size += ss
            if cur_size > self.chunksize:
                break
            ii -= 1
        else:
            return
        return ii

    def init_tracers(self):

        seeds = np.loadtxt(self.seed_path)
        self.tracers = []

        for num, *coords, it_init, t_init, weight in seeds:
            self.tracers.append(tracer(num=num,
                                       position=np.array(coords),
                                       t_init=t_init,
                                       to_trace=self.to_trace,
                                       weight=weight,
                                       termvar=self.terminate_var,
                                       termval=self.terminate_val,
                                       data_getter=self.get_data,
                                       max_step=self.max_step))
        if self.verbose:
            print(f"loaded {len(self.tracers)} tracer")
        return max(tr.times[-1] for tr in self.tracers)

    def get_data(self, tt, pos, keys):
        coords = {'x': pos[0], 'y': pos[1], 'z': pos[2]}

        ind = self.times.searchsorted(tt, side='right')

        data = {kk: np.array([self.dats[self.its[ii]][kk](**coords)[0]
                              for ii in range(ind-self.off, ind+self.off)])
                for kk in keys}
        result = np.array([interp1d(self.times[ind-self.off: ind+self.off],
                           data[kk], kind=self.t_int_kind)(tt)
                           for kk in keys])
        return result

    def integrate_all(self):
        while True:
            i_end = self.load_chunk(self.i_start)
            if i_end is None:
                break
            t_start = self.times[self.i_start-self.off+1]
            t_end = self.times[i_end+self.off]
            if self.verbose:
                print(f"integrating from t={t_start:.2f}M to t={t_end:.2f}M")

            for nn, tr in enumerate(self.tracers):
                if self.verbose:
                    print(f"integrating tracer {nn} ({tr.num})           ", end='\r')
                tr.integrate(t_start, t_end)

            n_not_started = sum(tr.status == -2 for tr in self.tracers)
            n_running = sum(tr.status == 0 for tr in self.tracers)
            n_failed = sum(tr.status == -1 for tr in self.tracers)
            n_done = sum(tr.status == 1 for tr in self.tracers)
            if n_running+n_not_started == 0:
                break
            if self.verbose:
                print(f"{n_not_started} not started, {n_running} running, {n_failed} failed, {n_done} done")
            self.i_start = i_end + self.off*2
        if self.verbose:
            print()
            print("Done")

    def save_all(self):
        for tr in self.tracers:
            tr.save(self.save_path)


class tracer():
    """
    A single Tracer object. Basically a wrapper around scipy.integrate.solve_ivp.
    """

    def __init__(self,
                 num,
                 position,
                 t_init,
                 to_trace,
                 weight,
                 data_getter,
                 max_step=203.025,
                 termvar=None,
                 termval=None,
                 ):

        self.num = int(num)
        self.max_step = max_step
        self.weight = weight
        self.get_data = data_getter

        self.termvar = termvar
        self.termval = termval

        self.keys = ['V^x', 'V^y', 'V^z']

        self.times = np.array([t_init])
        self.pos = np.array([position])
        self.trace = {kk: np.array([]) for kk in to_trace}
        if self.termvar not in to_trace:
            self.trace[self.termvar] = np.array([])

        self.t_step = None
        self.status = -2

    def get_rhs(self, tt, pos):
        try:
            return self.get_data(tt, pos, self.keys)
        except KeyError:
            print(tt)
            raise

    def set_trace(self, time, pos):
        self.times = np.concatenate((self.times, time))
        self.pos = np.concatenate((self.pos, pos))
        trace = np.array([self.get_data(tt, yy, self.trace.keys())
                          for tt, yy in zip(time, pos)])
        for kk, tt in zip(self.trace, trace.T):
            self.trace[kk] = np.concatenate((self.trace[kk], tt))

    def integrate(self, t_start, t_end):
        if self.status == 1 or self.status == -1:
            return
        min_time, max_time = min(t_start, t_end), max(t_start, t_end)
        if min_time <= self.times[-1] <= max_time:
            sol = sint.solve_ivp(self.get_rhs,
                                 (self.times[-1], t_end),
                                 self.pos[-1],
                                 first_step=self.t_step,
                                 rtol=1e-5,
                                 max_step=self.max_step)
            self.t_step = abs(sol.t[-2] - sol.t[-3])
            self.set_trace(sol.t, sol.y.T)

            self.status = sol.status
            if sol.status == -1:
                self.message = sol.message

            if self.termvar is not None:
                term = self.trace[self.termvar]
                if any(term >= self.termval) and any(term <= self.termval):
                    self.status = 1
        return self.status

    def save(self, path):
        self.times = self.times[1:]
        self.pos = self.pos[1:]
        self.times, inds = np.unique(self.times, return_index=True)
        out = np.stack((self.times, *self.pos[inds].T, *[val[inds] for val in self.trace.values()]))
        header = f"status={self.status}, mass={self.weight}\n"
        header += f"{'time':>13s} {'x':>15s} {'y':>15s} {'z':>15s} "
        fmt = "%15.7e "*4
        for kk in self.trace:
            fmt += "%15.7e "
            header += f"{kk:>15s} "
        np.savetxt(f"{path}/tracer_{self.num:07d}.dat", out.T, fmt=fmt, header=header)


def load_tracer(file_path):
    regex = "status=(.*), mass=(.*)"
    with open(file_path, 'r') as ff:
        header = ff.readline()
        header = header.replace('# ', '')
        ma = re.match(regex, header)
        if ma is not None:
            status, mass = ma.groups()
        else:
            status, mass = 0, 0
        header = ff.readline()
    header = header.replace('# ', '')
    keys = header.split()

    data = dict(zip(keys, np.loadtxt(file_path, unpack=True)))
    data['mass'] = float(mass)
    data['status'] = int(status)
    return data
