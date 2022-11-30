import re
from warnings import warn
from functools import reduce
from time import sleep
import numpy as np
import scipy.integrate as sint
from scipy.interpolate import interp1d
from .Utils import Units


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
            raise ValueError(
                f"Time interpolation with order {t_int_order} not supported")

        self.vars = {var: self.sim.get_variable(var)
                     for var in self.to_trace+('V^x', 'V^y', 'V^z')}

        all_its = [var.available_its(self.region) for var in self.vars.values()]
        self.its = reduce(np.intersect1d, all_its)
        self.times = self.sim.get_time(self.its)
        self.max_step = self.times[1] - \
            self.times[0] if max_step is None else max_step

        max_t = self.init_tracers()
        i_start = self.times.searchsorted(max_t, side='left') + self.off + 1
        self.times = self.times[:i_start]
        self.its = self.its[:i_start]

        self.i_start = len(self.times)-self.off

    def load_chunk(self, i_start):
        large_it = self.its[i_start]
        cur_size = 0
        for it, dat in self.dats.items():
            if it > large_it:
                del dat
            else:
                for dd in dat.values():
                    cur_size += sum(md.size *
                                    md.itemsize for md in dd.mem_data.values())
        self.dats = {}
        ii = i_start
        for it in self.its[i_start::-1]:
            if it not in self.dats:
                for n_try in range(4):
                    try:
                        self.dats[it] = {kk: var.get_data(region=self.region, it=it)
                                         for kk, var in self.vars.items()}
                        break
                    except OSError as ex:
                        if "Resource temporarily unavailable" in str(ex):
                            sleep(10)
                            continue
                        raise
                else:
                    raise OSError(
                        f"Could not open hdf5 file after {n_try} tries") from ex
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
                                       max_step=self.max_step,
                                       save_path=self.save_path))
        if self.verbose:
            print(f"loaded {len(self.tracers)} tracer", flush=True)
        return max(tr.times[-1] for tr in self.tracers)

    def get_data(self, tt, pos, keys):
        if tt < self.times.min() or tt > self.times.max():
            raise RuntimeError(
                f"Interpolation time {tt} not in loaded times {self.times.min()}:{self.times.max()}")
        elif not np.isfinite(tt):
            raise RuntimeError(f"Interpolation time is not finite: {tt}")

        coords = {'x': pos[0], 'y': pos[1], 'z': pos[2]}

        ind = self.times.searchsorted(tt, side='left')
        inds = np.arange(ind-self.off, ind+self.off)
        data = {}
        for it in self.its[inds]:
            if it not in self.dats:
                self.dats[it] = {kk: var.get_data(region=self.region, it=it)
                                 for kk, var in self.vars.items()}

        data = {kk: np.array([self.dats[it][kk](**coords)[0] for it in self.its[inds]])
                for kk in keys}

        result = np.array(
            [interp1d(self.times[inds], data[kk], kind=self.t_int_kind)(tt) for kk in keys])
        return result

    def integrate_all(self):
        while True:
            i_end = self.load_chunk(self.i_start)
            if i_end is None:
                for tr in self.tracers:
                    if tr.status == 0:
                        tr.save()
                break

            i_end_off = min(len(self.times)-1, i_end + self.off)
            i_start_off = min(len(self.times)-1, i_start - self.off)
            

            t_start = self.times[i_start_off]
            t_end = self.times[i_end_off]
            it_start = self.its[i_start_off]
            it_end = self.its[i_end_off]
            if self.verbose:
                print(
                    f"integrating from t={t_start:.2f}M to t={t_end:.2f}M (it={it_start} to it={it_end})", flush=True)

            for nn, tr in enumerate(self.tracers):
                if tr.status in [-1, 1]:
                    continue
                if self.verbose:
                    print(
                        f"integrating tracer {nn} ({tr.num})           ", end='\r', flush=True)
                for ntry in range(10):
                    try:
                        tr.integrate(t_start, t_end)
                        break
                    except KeyboardInterrupt:
                        raise
                    except OSError as ee:
                        os_ex = ee
                        sleep(30)
                        continue
                    except Exception as ee:
                        warn(
                            f"{tr.num} had an exception in integration step "
                            f"{t_start, t_end}\n{type(ee).__name__}: {str(ee)}"
                            )
                        tr.status = -1
                        tr.save()
                        break
                else:
                    warn(
                        f"{tr.num} had an OSError {ntry} times in integration step "
                        f"{t_start, t_end}\n{type(os_ex).__name__}: {str(os_ex)}"
                        )
                    tr.status = -1
                    # raise RuntimeError(f"OS error after {ntry} tries") from os_ex


            n_not_started = sum(tr.status == -2 for tr in self.tracers)
            n_running = sum(tr.status == 0 for tr in self.tracers)
            n_failed = sum(tr.status == -1 for tr in self.tracers)
            n_done = sum(tr.status == 1 for tr in self.tracers)
            if n_running+n_not_started == 0:
                break
            if self.verbose:
                print(
                    f"{n_not_started} not started, "
                    f"{n_running} running, "
                    f"{n_failed} failed, "
                    f"{n_done} done",
                    flush=True)
                if 'temp' in self.to_trace:
                    temps = [tr.trace['temp'][-1]
                             for tr in self.tracers
                             if len(tr.trace['temp']) > 0]
                    if len(temps)>0:
                        print(
                            f"temperature range: "
                            f"{min(temps)*11.604518121745585:.1f}, "
                            f"{max(temps)*11.604518121745585:.1f}"
                        )
            self.i_start = i_end + self.off*2
        if self.verbose:
            print("Done", flush=True)
        return np.array([tr.status for tr in self.tracers])


class tracer():
    """
    A single Tracer object.
    Basically a wrapper around scipy.integrate.solve_ivp.
    """

    def __init__(self,
                 num,
                 position,
                 t_init,
                 to_trace,
                 weight,
                 data_getter,
                 save_path,
                 max_step=203.025,
                 termvar=None,
                 termval=None,
                 verbose=False,
                 ):

        self.num = int(num)
        self.max_step = max_step
        self.weight = weight
        self.get_data = data_getter
        self.save_path = save_path

        self.termvar = termvar
        self.termval = termval
        self.verbose = verbose

        self.keys = ['V^x', 'V^y', 'V^z']

        self.times = np.array([t_init])
        self.pos = np.array([position])
        self.trace = {kk: np.array([]) for kk in to_trace}
        if self.termvar not in to_trace:
            self.trace[self.termvar] = np.array([])

        self.t_step = None
        self.status = -2

    def get_rhs(self, tt, pos):
        ex = RuntimeError("This should never happen")
        for n_try in range(5):
            try:
                return self.get_data(tt, pos, self.keys)
            except KeyboardInterrupt:
                raise
            except Exception as ee:
                ex = ee
                if self.verbose:
                    print(
                        f"Tracer {self.num}: get_rhs had exception {ex}, {n_try+1} tries"
                    )
                sleep(5)
                continue
        raise ex

    def set_trace(self, time, pos):
        self.times = np.concatenate((self.times, time[1:]))
        self.pos = np.concatenate((self.pos, pos[1:]))
        trace = np.array([self.get_data(tt, yy, self.trace.keys())
                          for tt, yy in zip(time, pos)])
        for kk, tt in zip(self.trace, trace.T):
            self.trace[kk] = np.concatenate((self.trace[kk], tt[1:]))

    def integrate(self, t_start, t_end):
        if self.status == 1 or self.status == -1:
            return self.status
        min_time, max_time = min(t_start, t_end), max(t_start, t_end)
        if np.isclose(max_time, self.times[-1]):
            self.times[-1] = max_time*0.9999
        if min_time <= self.times[-1] <= max_time:
            if self.t_step is not None:
                if self.t_step > (max_t_step := np.abs(self.times[-1] - t_end)):
                    self.t_step = max_t_step
                if self.t_step <= 0:
                    self.t_step = None

            sol = sint.solve_ivp(self.get_rhs,
                                 (self.times[-1], t_end),
                                 self.pos[-1],
                                 first_step=self.t_step,
                                 rtol=1e-5,
                                 max_step=self.max_step)
            if len(sol.t) > 2:
                self.t_step = np.abs(sol.t[-2] - sol.t[-3])
            elif len(sol.t) == 2:
                self.t_step = np.abs(sol.t[-1] - sol.t[-2])
            self.set_trace(sol.t, sol.y.T)

            if 'temp' in self.trace:
                if np.all(self.trace['temp']>10/11.605) and len(self.trace['temp']) > 20:
                    self.status = -1
                    self.message = "More then 20 timesteps with temp>10GK"
                    self.save()
                    return self.status

            self.status = sol.status
            if sol.status == -1:
                self.message = sol.message
                self.save()
                return self.status


            if self.termvar is not None:
                term = self.trace[self.termvar]
                if any(term >= self.termval) and any(term <= self.termval):
                    self.status = 1
                    self.save()
        return self.status

    def save(self):
        if len(self.times) < 2:
            return
        self.times = self.times[1:]
        self.pos = self.pos[1:]
        _, uinds = np.unique(np.round(self.times, 7), return_index=True)
        utimes = self.times[uinds]

        t0 = utimes.min()
        utimes -= t0

        utimes *= Units['Time']
        t0 *= Units['Time']
        self.pos *= Units['Length']
        if 'rho' in self.trace:
            self.trace['rho'] *= Units['Rho']
        if 'temp' in self.trace:
            self.trace['temp'] *= 11.604518121745585

        out = np.stack((utimes, *self.pos[uinds].T,
                        *[val[uinds] for val in self.trace.values()]))
        header = f"status={self.status}, mass={self.weight}, t0={t0}\n"
        if self.status < 0:
            header = f"message={self.message}\n"
        header += f"{'time':>13s} {'x':>15s} {'y':>15s} {'z':>15s} "
        fmt = "%15.7e "*4
        for kk in self.trace:
            fmt += "%15.7e "
            header += f"{kk:>15s} "
        np.savetxt(f"{self.save_path}/tracer_{self.num:07d}.dat",
                   out.T, fmt=fmt, header=header)


class Trajectory:
    def __init__(self, file_path):
        regex = "status=(.*), mass=([0-9\.e+-]+)(, t0=)?(.*)?"
        with open(file_path, 'r') as ff:
            header = ff.readline()
            header = header.replace('# ', '')
            ma = re.match(regex, header)
            if ma is not None:
                status, mass, _, t0 = ma.groups()
                if t0 == '':
                    t0 = 0
            else:
                raise RuntimeError("Could not read header:\n"+header)
            header = ff.readline()
        header = header.replace('# ', '')
        keys = header.split()

        for key, data in zip(keys, np.loadtxt(file_path, unpack=True)):
            setattr(self, key, data)
            # = dict(zip(keys, np.loadtxt(file_path, unpack=True)))
        self.mass = float(mass)
        self.status = int(status)
        self.t0 = float(t0)
        if len(nums := re.findall(r'\d+', file_path)) > 0:
            self.num = int(nums[-1])
