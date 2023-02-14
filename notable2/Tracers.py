import re
from warnings import warn
from functools import reduce
from time import sleep
import numpy as np
import scipy.integrate as sint
from scipy.interpolate import interp1d
from typing import Optional
from numpy.typing import NDArray

from .Utils import Units
from .Simulation import Simulation

NOT_STARTED = -2
CRASHED = -1
RUNNING = 0
FINISHED = 1

MeV_to_GK = 11.604518121745585


class TracerBunch():
    """
    A bunch of tracers. Used to coordinate the calculation of multiple tracers in bunches based on memory allocation.

    Arguments:
        sim: Simulation object
        save_path: (str) path to save the data
        seed_path: (str) path to the seed file
        max_step: (float) maximum step size for the tracers
        to_trace: (tuple) variables to trace
        verbose: (bool) print progress
        terminate_var: (str) variable to terminate the tracer on
        terminate_val: (float) value of the variable to terminate the tracer on
        t_int_order: (int) order of the time interpolation

    Methods:
        init_tracers: initialize the tracers
        load_chunk: load the next chunk of data
        get_data: interpolate data at the position and time of a tracer
        integrate_all: integrate all tracers
    """

    def __init__(self,
                 sim: Simulation,
                 save_path: str,
                 seed_path: str,
                 max_step: Optional[np.float_] = None,
                 to_trace: tuple[str, ...] = ('rho', 'temp', 'ye'),
                 verbose: bool = False,
                 terminate_var: Optional[str] = None,
                 terminate_val: Optional[np.float_] = None,
                 t_int_order: int = 1,
                 ):

        self.sim = sim
        self.seed_path = seed_path
        self.save_path = save_path
        self.to_trace = tuple(to_trace)
        self.verbose = verbose
        self.seed_path = seed_path
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

        all_its = [var.available_its(self.region)
                   for var in self.vars.values()]
        self.its = reduce(np.intersect1d, all_its)
        self.times = np.round(self.sim.get_time(
            self.its), 3)  # round for comparissons
        self.max_step = self.times[1] - \
            self.times[0] if max_step is None else max_step

        max_t = self.init_tracers()

        assert max_t <= self.times.max(
        ), f"Maximum tracer time {max_t} is larger then simulations times {self.times.max()}"

        self.cur_ind = self.times.searchsorted(max_t, side='left')
        self.times = self.times[:self.cur_ind + self.off]
        self.its = self.its[:self.cur_ind + self.off]

    def message(self, msg, **kwargs):
        if self.verbose:
            print(msg, flush=True, **kwargs)

    def load_chunk(self):
        """
        Load the next chunk of data
        """
        cur_its = self.its[self.cur_ind-self.off: self.cur_ind+self.off]
        # delete old data
        for it, dat in self.dats.items():
            if it not in cur_its:
                del dat

        # load new data
        for it in cur_its:
            if it not in self.its or it in self.dats:
                continue

            self.dats[it] = {}
            for kk, var in self.vars.items():
                # try 5 times and wait if file is unavailable
                for n_try in range(5):
                    try:
                        self.dats[it][kk] = var.get_data(
                            region=self.region, it=it
                        )
                        break
                    except BlockingIOError:
                        sleep(10)
                else:
                    raise RuntimeError(
                        f"Could not open {kk} hdf5 file after {n_try} tries"
                    )

            # set save in memory flag to true
            for kk in self.dats[it]:
                self.dats[it][kk].mem_load = True

    def init_tracers(self):
        """
        Initialize the tracers
        """
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

        self.message(f"loaded {len(self.tracers)} tracer")

        return max(tr.times.max() for tr in self.tracers)

    def get_data(self, tt, pos, keys):
        """
        Interpolate the data for 'keys' at time 'tt' and position 'pos'
        """
        if tt < self.times.min() or tt > self.times.max():
            raise RuntimeError(f"Interpolation time {tt} not in loaded times "
                               f"{self.times.min()}:{self.times.max()}")
        elif not np.isfinite(tt):
            raise RuntimeError(f"Interpolation time is not finite: {tt}")

        coords = {'x': pos[0], 'y': pos[1], 'z': pos[2]}

        ind = self.times.searchsorted(tt, side='left')
        inds = np.arange(ind-self.off, ind+self.off)
        data = {}

        for it in self.its[inds]:
            if it not in self.dats:
                raise RuntimeError(f"it {it} not loaded. "
                                   f"Loaded its: {list(self.dats.keys())} "
                                   "This should not happen")
            for kk in keys:
                data[kk] = self.dats[it][kk](**coords)[0]

        result = np.array([
            interp1d(
                self.times[inds],
                data[kk],
                kind=self.t_int_kind
            )(tt) for kk in keys])
        return result

    def integrate_all(self):
        """
        Integrate all tracers
        """
        while self.cur_ind > 0:

            self.load_chunk()

            t_start = self.times[self.cur_ind]
            t_end = self.times[self.cur_ind - 1]
            it_start = self.its[self.cur_ind]
            it_end = self.its[self.cur_ind - 1]

            self.message(f"integrating from t={t_start:.2f}M to t={t_end:.2f}M "
                         f"(it={it_start} to it={it_end})")

            for nn, tr in enumerate(self.tracers):
                self.message(f"integrating tracer {nn} ({tr.num})           ",
                             end='\r')
                # try integrating tracers
                try:
                    tr.integrate(t_start, t_end)
                except KeyboardInterrupt:
                    raise
                except Exception as ee:
                    raise
                    # if integration fails mark tracer as crashed and save
                    tr.message = (f"Error in t={t_start}-{t_end}\n"
                                  f"{type(ee).__name__}: {str(ee)}")
                    warn(f"Tr. {tr.num}: {tr.message}")
                    tr.status = CRASHED
                    tr.save()

            n_not_started, n_running, n_finished, n_crashed = 0, 0, 0, 0
            for tr in self.tracers:
                if tr.status == NOT_STARTED:
                    n_not_started += 1
                elif tr.status == RUNNING:
                    n_running += 1
                elif tr.status == FINISHED:
                    n_finished += 1
                elif tr.status == CRASHED:
                    n_crashed += 1

            # if all tracers are finished or crashed, we are done
            if n_running+n_not_started == 0:
                break

            # advance index backward in time
            self.cur_ind = self.cur_ind - 1

            if not self.verbose:
                continue
            self.message(f"{n_not_started} not started, "
                         f"{n_running} running, "
                         f"{n_crashed} failed, "
                         f"{n_finished} done",)

            # print some status messages about current temperature and position
            if 'temp' in self.to_trace:
                temps = [tr.trace['temp'][-1] for tr in self.tracers
                         if tr.status == RUNNING and len(tr.trace['temp']) > 0]
                if len(temps) > 0:
                    self.message(f"T = "
                                 f"{min(temps)*MeV_to_GK:.1f}-"
                                 f"{max(temps)*MeV_to_GK:.1f}GK",
                                 end=" | ")

            radii = [np.linalg.norm(tr.pos[-1]) for tr in self.tracers
                     if tr.status == RUNNING and len(tr.pos) > 0]
            if len(radii) > 0:
                self.message(f"r = "
                             f"{min(radii)*Units['Length']:.0f}-"
                             f"{max(radii)*Units['Length']:.0f}km")

        # save all tracers that have not met any stopping criterion
        for tr in self.tracers:
            if tr.status == RUNNING:
                tr.save()

        self.message("")
        self.message("Done")
        return np.array([tr.status for tr in self.tracers])


class tracer():
    """
    A single Tracer object.
    Basically just a wrapper around scipy.integrate.solve_ivp.
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

        self.t_step = None
        self.status = NOT_STARTED
        self.saved = False

        self.keys = ['V^x', 'V^y', 'V^z']

        self.times = np.array([t_init])
        self.pos = np.array([position])
        self.trace = {kk: np.array([]) for kk in to_trace}
        if self.termvar not in to_trace:
            self.trace[self.termvar] = np.array([])

        # trace = self.get_data(t_init, position, self.trace.keys())
        # for kk, tt in zip(self.trace, trace.T):
        #     self.trace[kk] = np.array([tt])

    def get_rhs(self, tt, pos):
        """
        Get the right hand side of the ODE system.
        Just a wrapper for the parent Tracer bu
        """
        return self.get_data(tt, pos, self.keys)

    def set_trace(self, time, pos):
        """
        Set the current position and time of the tracer and interpolate all data
        in self.trace.
        """
        self.times = np.concatenate((self.times, time[1:]))
        self.pos = np.concatenate((self.pos, pos[1:]))
        trace = np.array([self.get_data(tt, yy, self.trace.keys())
                          for tt, yy in zip(time, pos)])
        for kk, tt in zip(self.trace, trace.T):
            self.trace[kk] = np.concatenate((self.trace[kk], tt[1:]))

    def integrate(self, t_start, t_end):
        """
        Integrate the tracer from t_start to t_end.
        """
        if self.status == FINISHED or self.status == CRASHED:
            return self.status

        if t_start < self.times[-1] and self.status == NOT_STARTED:
            raise ValueError(f"Tracer {self.num}'s time has passed but "
                             "it was never started. Maybe seed time is too "
                             "late?")

        if t_end >= self.times[-1]:
            return self.status

        if self.t_step is None:
            pass
        elif self.t_step > (max_t_step := np.abs(self.times[-1] - t_end)):
            self.t_step = max_t_step
        elif self.t_step <= 0:
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

        radii = np.linalg.norm(self.pos, axis=-1)
        dt = (self.times.max() - self.times.min())*Units['Time']
        if dt > 5 and radii.max()-radii.min() < radii.max()*1e-2:
            self.message = "Not moving for 5ms"
            self.status = CRASHED
            warn(f"Tr. {self.num}: {self.message}")
            self.save()
            return self.status
        if np.any(radii < 100):
            self.message = "Radius < 100km"
            self.status = CRASHED
            warn(f"Tr. {self.num}: {self.message}")
            self.save()
            return self.status

        self.status = sol.status
        if sol.status == CRASHED:
            self.message = sol.message
            self.save()
            return self.status

        if self.termvar is not None:
            term = self.trace[self.termvar]
            if any(term >= self.termval):  # and any(term <= self.termval):
                self.status = FINISHED
                self.save()
        return self.status

    def save(self):
        """
        Format tracer output and save the tracer to a file.
        """
        # if tracer is allready saved nothing needs to be done
        if self.saved:
            return

        # format units
        self.times *= Units['Time']
        self.pos *= Units['Length']
        if 'rho' in self.trace:
            self.trace['rho'] *= Units['Rho']
        if 'temp' in self.trace:
            self.trace['temp'] *= MeV_to_GK

        # subtract starting time for better floating point accuracy in text file
        t0 = self.times.min()
        self.times -= t0

        # resort arrays based unique sorted times
        _, uinds = np.unique(np.round(self.times, 3), return_index=True)
        utimes = self.times[uinds]
        out = np.stack((utimes, *self.pos[uinds].T,
                        *[val[uinds] for val in self.trace.values()]))

        # write to file
        header = f"status={self.status}, mass={self.weight}, t0={t0}\n"
        if self.status == CRASHED:
            header += f"message={self.message}\n"
        header += f"{'time':>13s} {'x':>15s} {'y':>15s} {'z':>15s} "

        fmt = "%15.7e "*4
        for kk in self.trace:
            fmt += "%15.7e "
            header += f"{kk:>15s} "
        np.savetxt(f"{self.save_path}/tracer_{self.num:07d}.dat",
                   out.T, fmt=fmt, header=header)
        self.saved = True


class Trajectory:
    """
    Class for a trajectory of a tracer particle.

    Attributes:
        num (int): The number of the tracer.
        weight (float): The mass of the tracer.
        times (np.ndarray): The times of the tracer.
        pos (np.ndarray): The positions of the tracer.
        trace (dict): The data of the tracer.
        status (int): The status of the integration.

    Methods:
        get_data (tt, pos, keys): Get the data of the tracer at a given time and position.
    """
    x: NDArray[np.float_]
    y: NDArray[np.float_]
    z: NDArray[np.float_]
    time: NDArray[np.float_]
    rho: NDArray[np.float_]
    temp: NDArray[np.float_]
    ye: NDArray[np.float_]
    t0: float
    mass: float
    num: int
    status: int

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

        self.mass = float(mass)
        self.status = int(status)
        self.t0 = float(t0)
        if len(nums := re.findall(r'\d+', file_path)) > 0:
            self.num = int(nums[-1])

        with open(file_path, 'r') as ff:
            for _ in range(10):
                nheader = ff.readline()
                if nheader[0] != '#':
                    break
                header = nheader
            else:
                raise RuntimeError(
                    f"{file_path} does not have header")

        header = header.replace('# ', '')
        keys = header.split()

        for key, data in zip(keys, np.loadtxt(file_path, unpack=True)):
            if isinstance(data, float):
                data = np.array([data])
            setattr(self, key, data)

        if hasattr(self, 'time'):
            self.time += self.t0

    def get_at(self, key: str, val: float, target: str, extend=False):
        """
        Get the data for 'target' where 'key' has value 'val
        """
        # Check if key and target exist
        if not hasattr(self, key):
            raise ValueError(f"{key} not in this trajectory")
        if not hasattr(self, target):
            raise ValueError(f"{target} not in this trajectory")

        # get the data for the key and target
        val_ar = getattr(self, key)
        target_ar = getattr(self, target)

        # check if val is in the array
        if not extend:
            if val_ar.min() > val:
                raise ValueError(f"{key} is allways larger then {val}")
            if val_ar.max() < val:
                raise ValueError(f"{key} is allways smaller then {val}")
        else:
            if val_ar.min() > val:
                return target_ar[np.argmin(val_ar)]
            if val_ar.max() < val:
                return target_ar[np.argmax(val_ar)]

        # do the interpolation
        return interp1d(val_ar, target_ar, )(val)
