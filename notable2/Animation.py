from abc import ABC, abstractmethod
from functools import reduce
from inspect import signature
from typing import Iterable, Callable, Optional, Union, TYPE_CHECKING, Any
import matplotlib.pyplot as plt  # type: ignore
import numpy as np
from numpy.typing import NDArray
from matplotlib.animation import FuncAnimation  # type: ignore

from .Utils import RLArgument, Units, func_dict, Plot2D
from .Plot import _handle_kwargs, _handle_PPkwargs
from .Variable import Variable

if TYPE_CHECKING:
    from .Utils import Simulation


class AniFunc(ABC):

    ani: "Animation"

    @abstractmethod
    def get_times(self,
                  min_time: Optional[float],
                  max_time: Optional[float],
                  every: int):
        ...

    @abstractmethod
    def init(self, fig):
        ...

    @abstractmethod
    def __call__(self, time: np.float_):
        ...


class Animation:

    fig: plt.Figure
    funcs: list[AniFunc]
    init_func: Optional[Callable]
    times: NDArray[np.float_]
    min_time: Optional[float]
    max_time: Optional[float]
    every: int

    def __init__(self,
                 min_time: Optional[float] = None,
                 max_time: Optional[float] = None,
                 every: int = 1,
                 init_func: Optional[Callable] = None,
                 ):
        self.funcs = []
        self.times = np.array([], dtype=float)
        self.min_time = min_time
        self.max_time = max_time
        self.every = every
        self.init_func = init_func

    def add_animation(self, func: AniFunc):
        self.funcs.append(func)
        func.ani = self
        times = func.get_times(min_time=self.min_time,
                               max_time=self.max_time,
                               every=self.every)
        self.times = np.unique(np.append(self.times, times))
        self.times.sort()

    def animate(self, fig: plt.Figure, **kwargs) -> FuncAnimation:
        def _init():
            for func in self.funcs:
                func.init()
            if self.init_func is not None:
                self.init_func()

        def _animate(time: np.float_):
            for func in self.funcs:
                func(time)

        ani = FuncAnimation(fig=fig, frames=self.times, func=_animate, init_func=_init, **kwargs)
        plt.close(fig)
        return ani


class GDAniFunc(AniFunc):

    sim: "Simulation"
    key: str
    var: Variable
    its: NDArray[np.int_]
    times: NDArray[np.float_]
    image: Union[Plot2D, plt.Line2D]
    # -----------Plot kwargs-------------------------------
    rls: NDArray[np.int_]
    region: str
    setup_at: float
    code_units: bool
    exclude_ghosts: int = 0
    label: Union[bool, str]
    title: Union[bool, str]
    xlabel: Union[bool, str]
    ylabel: Union[bool, str]
    # -----------Variable kwargs----------------------------
    func: Optional[Union[Callable]] = None
    slice_ax: Optional[dict[str, float]] = None
    interp_ax: Optional[dict[str, float]] = None
    # -----------kwargs dicts----------------------------
    kwargs: dict[str, Any]
    PPkwargs: dict[str, Any]

    def __init__(self,
                 sim: "Simulation",
                 key: str,
                 # -----------Plot kwargs-------------------------------
                 rls: "RLArgument" = None,
                 region: Optional[str] = None,
                 setup_at: Union[int, float] = 0.,
                 code_units: bool = False,
                 exclude_ghosts: int = 0,
                 label: Union[bool, str] = False,
                 title: Union[bool, str] = True,
                 xlabel: Union[bool, str] = True,
                 ylabel: Union[bool, str] = True,
                 # -----------Variable kwargs----------------------------
                 func: Optional[Union[Callable, str, bool]] = None,
                 slice_ax: Optional[dict[str, float]] = None,
                 interp_ax: Optional[dict[str, float]] = None,
                 # ------------------------------------------------------
                 **kwargs):
        self.sim = sim
        self.key = key
        self.var = sim.get_variable(key)

        if region is None:
            self.region = 'xz' if self.sim.is_cartoon else 'xy'
        else:
            self.region = region

        self.code_units = code_units
        self.exclude_ghosts = exclude_ghosts
        self.setup_at = float(setup_at)

        var_kwargs, popped = _handle_kwargs(self.var.kwargs, dict(func=(func, None),
                                                                  slice_ax=(slice_ax, None),
                                                                  interp_ax=(interp_ax, None)))
        self.slice_ax = popped["slice_ax"]
        self.interp_ax = popped["interp_ax"]
        func = popped["func"]

        func_str: Optional[str]
        if isinstance(func, str):
            func_str, self.func = func_dict[func]
        else:
            func_str = None

        self.kwargs = {**var_kwargs, **kwargs}

        _, self.PPkwargs = _handle_PPkwargs(self.kwargs, self.var)

        self.rls = sim.expand_rl(rls)

        pn_str = self.var.plot_name.print(code_units=code_units, **self.PPkwargs)
        if func_str is not None:
            pn_str = func_str.format(pn_str)
        if title is True:
            title = f"{pn_str}\nTIME"
        if isinstance(title, str):
            self.title = title.replace('PLOTNAME', pn_str)
            self.title = self.title.replace('SIM', sim.nice_name)
        else:
            self.title = title

        self.label = self.sim.nice_name if label is True else label

        self.xlabel = xlabel
        self.ylabel = ylabel

    def get_times(self, min_time, max_time, every):

        its = self.var.available_its(self.region)
        times = self.sim.get_time(its)
        if self.sim.t_merg is not None:
            times -= self.sim.t_merg

        mask = np.ones_like(its, dtype=bool)
        if min_time is not None:
            mask = mask & (times >= min_time/(Units['Time'] if not self.code_units else 1))
        if max_time is not None:
            mask = mask & (times <= max_time/(Units['Time'] if not self.code_units else 1))
        self.its = its[mask][::every]
        self.times = times[mask][::every]
        return self.times

    def init(self):
        init_it = int(self.its[int((len(self.its)-1)*self.setup_at)])

        self.image = self.sim.plotGD(key=self.key,
                                     it=init_it,
                                     region=self.region,
                                     rls=self.rls,
                                     code_units=self.code_units,
                                     title=self.title,
                                     label=self.label,
                                     xlabel=self.xlabel,
                                     ylabel=self.ylabel,
                                     func=self.func,
                                     slice_ax=self.slice_ax,
                                     interp_ax=self.interp_ax,
                                     exclude_ghosts=self.exclude_ghosts,
                                     **self.kwargs)

        self.kwargs, _ = _handle_PPkwargs(self.kwargs, self.var)

    def __call__(self, time: np.float_):
        if time > max(self.times):
            if self.region == 1:
                self.image.set_data([], [])
        if time not in self.times:
            return
        ii = self.times.searchsorted(time)
        it = self.its[ii]

        grid_func = self.var.get_data(region=self.region,
                                      it=it,
                                      exclude_ghosts=self.exclude_ghosts,
                                      **self.PPkwargs)
        coords = grid_func.coords
        if not self.code_units:
            data = {rl: grid_func.scaled(rl) for rl in self.rls}
            coords = {rl: {ax: cc*Units['Length']
                           for ax, cc in coords[rl].items()}
                      for rl in self.rls}
        else:
            data = {rl: grid_func[rl] for rl in self.rls}

        if callable(self.func):
            if isinstance(self.func, np.ufunc):
                data = {rl: self.func(dd) for rl, dd in data.items()}
            elif len(signature(self.func).parameters) == 1:
                data = {rl: self.func(dd, **self.PPkwargs) for rl, dd in data.items()}
            else:
                coords, data = {rl: self.func(dd, **coords[rl]) for rl, dd in data.items()}
        if len(self.region) == 1:
            dat = data[self.rls[-1]]
            xx = coords[self.rls[-1]][self.region]
            for rl in self.rls[-2::-1]:
                dat_rl = data[rl]
                x_rl = coords[rl][self.region]
                mask = (x_rl < xx.min()) | (x_rl > xx.max())
                xx = np.concatenate([xx, x_rl[mask]])
                dat = np.concatenate([dat, dat_rl[mask]])
            isort = np.argsort(xx)
            xx = xx[isort]
            dat = dat[isort]

            self.image.set_data(xx, dat)
        elif len(self.region) == 2:
            self.image.set_data(coords, data)

        if self.code_units:
            t_str = f"{time: .2f} $M_\\odot$"
        else:
            t_str = f"{time*Units['Time']: .2f} ms"
        if self.sim.t_merg is not None:
            t_str = f"$t - t_{{\\rm merg}}$ = {t_str}"
        else:
            t_str = f"$t$ = {t_str}"
        new_title = self.title
        if isinstance(new_title, str):
            new_title = new_title.replace('TIME', t_str)
            new_title = new_title.replace('IT', f'{it}')
            self.image.axes.set_title(new_title)


class TSLineAniFunc(AniFunc):

    def __init__(self,
                 sim: "Simulation",
                 key: str,
                 ax: plt.Axes,
                 code_units: bool,
                 **kwargs):
        self.sim = sim
        self.ax = ax
        self.kwargs = kwargs
        self.key = key
        self.code_units = code_units

    def get_times(self, *_):
        return np.array([], dtype=float)

    def init(self, fig):
        self.im = self.ax.axvline(0, **self.kwargs)

    def __call__(self, time: np.float_):
        self.im.set_xdata(time)
