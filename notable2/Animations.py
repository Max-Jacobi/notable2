from abc import ABC, abstractmethod
from inspect import signature
from typing import Callable, Optional, TYPE_CHECKING, Any, List, Dict
from tqdm import tqdm

import matplotlib.pyplot as plt  # type: ignore
import numpy as np
from matplotlib.animation import FuncAnimation  # type: ignore

from .Plot import _handle_kwargs, _handle_PPkwargs
from .Utils import Units, func_dict, Plot2D
from .Variable import Variable


if TYPE_CHECKING:
    from .Utils import Simulation, RLArgument
    from numpy.typing import NDArray


class AniFunc(ABC):
    "Abstract base class for animation functions."

    ani: "Animation"

    @abstractmethod
    def _get_times(self,
                   min_time: Optional[float],
                   max_time: Optional[float],
                   every: int):
        ...

    @abstractmethod
    def _init(self):
        ...

    @abstractmethod
    def __call__(self, time: np.float_):
        ...


class Animation:
    """
    Class for managing animations in matplotlib.

    Arguments for the constructor:
        min_time: Optional[float]
            The starting time for the animation. If None the start of the
            simulation is chosen. Defaults to None
        max_time: Optional[float]
            The end time for the animation. If None the end of the simulation is
            chosen. Defaults to None
        every: int
            An interval to animate every `every` time steps, defaults to 1
        init_func: Optional[Callable]
            An optional function to call once at the beginning of the animation

    Methods
        add_animation(self, func: AniFunc)
            Adds an animation function to the list of functions to animate
        animate(fig: plt.Figure, **kwargs) -> FuncAnimation
            Animate the functions in the figure and returns a FuncAnimation object.
        save_frames(fig: plt.Figure,
                    path: str,
                    file_format: str = 'png',
                    verbose: bool = True,
                    **kwargs)
            Save frameas of animation as picture files.
    """

    init_func: Optional[Callable]
    times: 'NDArray[np.float_]'
    min_time: Optional[float]
    max_time: Optional[float]
    every: int
    funcs: List[AniFunc]

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
        """
        Add an animation to the Animation class.

        Arguments:
            func:
                A object of type `AniFunc` that represents the animation to be
                added.
        """
        self.funcs.append(func)
        func.ani = self
        times = func._get_times(min_time=self.min_time,
                                max_time=self.max_time,
                                every=self.every)
        self.times = np.unique(np.append(self.times, times))
        self.times.sort()

    def animate(self, fig: plt.Figure, **kwargs) -> FuncAnimation:
        """
        Create the animation based on the added functions and return a
        FuncAnimation object.
        """
        def _init():
            for func in self.funcs:
                func._init()
            if self.init_func is not None:
                self.init_func()

        def _animate(time: np.float_):
            for func in self.funcs:
                func(time)

        ani = FuncAnimation(fig=fig, frames=self.times,
                            func=_animate, init_func=_init, **kwargs)
        return ani

    def save_frames(self,
                    fig: plt.Figure,
                    path: str,
                    file_format: str = 'png',
                    verbose: bool = True,
                    **kwargs):
        """
        Save all frames of the animation based on the added functions.
        The files with be saved as <path>_<frame_number>.<file_format>.
        """
        for func in self.funcs:
            func._init()
        if self.init_func is not None:
            self.init_func()

        def _animate(time: np.float_):
            for func in self.funcs:
                func(time)

        for ii, time in tqdm(
            enumerate(self.times),
            disable=not verbose,
            total=len(self.times)
        ):
            _animate(time)
            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.savefig(f'{path}_{ii}.{file_format}', **kwargs)


class GDAniFunc(AniFunc):
    sim: "Simulation"
    key: str
    var: Variable
    its: 'NDArray[np.int_]'
    times: 'NDArray[np.float_]'
    image: (Plot2D | plt.Line2D)
    ax: plt.Axes
    # -----------Plot kwargs-------------------------------
    rls: "RLArgument"
    region: str
    setup_at: float
    code_units: bool
    exclude_ghosts: int = 0
    label: (bool | str)
    title: (bool | str)
    xlabel: (bool | str)
    ylabel: (bool | str)
    # -----------Variable kwargs----------------------------
    func: Optional[(Callable | bool)] = None
    # -----------kwargs dicts----------------------------
    kwargs: Dict[str, Any]
    PPkwargs: Dict[str, Any]
    """
    Class animating GridFunction data

    For the use with notable2.Animations.Animation.

    Arguments:
        sim: Simulation
            The simulation to animate
        key: str
            The key of the variable to animate.
        rls: Optional[RLArgument]
            The refinement levels to animate. If None all refinement levels
            are animated. Defaults to None
        region: Optional[RLArgument]
            The region to animate. If None the region is the xy-plane
            Defaults to None
        setup_at: float between 0 and 1
            The fraction of the animation to setup the plot at. Defaults to
            0, i.e. the first itteration
        code_units: bool
            If True the data is converted to code units. Defaults to False
        exclude_ghosts: int
            The number of ghost cells to exclude from the plot. Defaults to
            0
        label: (bool | str)
            If True the label is set to the variable name. If a string is
            given the label is set to that string. Defaults to False
        title: (bool | str)
            If True the title is set to the variable name. If a string is
            given the title is set to that string. Defaults to True
        xlabel: (bool | str)
            If True the xlabel is set to the x-axis name. If a string is
            given the xlabel is set to that string. Defaults to True
        ylabel: (bool | str)
            If True the ylabel is set to the y-axis name. If a string is
            given the ylabel is set to that string. Defaults to True
        ax: Optional[plt.Axes]
            The axes to plot on. If None a new figure is created. Defaults
            to None
        func: Optional[(Callable | str | bool)]
            If a callable is given it is used to transform the data. If a
            string is given it is used as a key to the `func_dict`
            dictionary in the `Variable` class. If True the default function
            for the variable is used. Defaults to None
        **kwargs:
            Keyword arguments to pass to the plotGD function on
            initialization
    """

    def __init__(self,
                 sim: "Simulation",
                 key: str,
                 # -----------Plot kwargs-------------------------------
                 rls: "RLArgument" = None,
                 region: Optional[str] = None,
                 setup_at: (int | float) = 0.,
                 code_units: bool = False,
                 exclude_ghosts: int = 0,
                 label: (bool | str) = False,
                 title: (bool | str) = True,
                 xlabel: (bool | str) = True,
                 ylabel: (bool | str) = True,
                 ax: Optional[plt.Axes] = None,
                 # -----------Variable kwargs----------------------------
                 func: Optional[(Callable | str | bool)] = None,
                 # ------------------------------------------------------
                 ** kwargs):
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
        if ax is None:
            self.ax = plt.gca()
        else:
            self.ax = ax

        var_kwargs, popped = _handle_kwargs(
            self.var.kwargs, dict(func=(func, None))
        )

        func = popped["func"]

        func_str: Optional[str]
        if isinstance(func, str):
            func_str, self.func = func_dict[func]
        else:
            self.func = func
            func_str = None

        self.kwargs = {**var_kwargs, **kwargs}

        _, self.PPkwargs = _handle_PPkwargs(self.kwargs, self.var)

        pn_str = self.var.plot_name.print(
            code_units=code_units, **self.PPkwargs)
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

        self.rls = rls

    def _get_times(self,
                   min_time: Optional[float],
                   max_time: Optional[float],
                   every: int,
                   ):

        its = self.var.available_its(self.region)
        times = self.sim.get_time(its)
        if self.sim.t_merg is not None:
            times -= self.sim.t_merg

        mask = np.ones_like(its, dtype=bool)
        if min_time is not None:
            mask = mask & (times >= min_time)
        if max_time is not None:
            mask = mask & (times <= max_time)
        self.its = its[mask][::every]
        self.times = np.round(times[mask][::every], -1)
        return self.times

    def _init(self):
        init_it = int(self.its[int((len(self.its)-1)*self.setup_at)])

        rls = self.sim.expand_rl(self.rls, it=init_it)
        self.image = self.sim.plotGD(key=self.key,
                                     it=init_it,
                                     region=self.region,
                                     rls=rls,
                                     code_units=self.code_units,
                                     title=self.title,
                                     label=self.label,
                                     xlabel=self.xlabel,
                                     ylabel=self.ylabel,
                                     func=self.func,
                                     ax=self.ax,
                                     exclude_ghosts=self.exclude_ghosts,
                                     **self.kwargs)

        self.kwargs, _ = _handle_PPkwargs(self.kwargs, self.var)

    def __call__(self, time: np.float_):
        if time > max(self.times):
            if len(self.region) == 1:
                self.image.set_data([], [])
                return
        if time not in self.times:
            return
        ii = self.times.searchsorted(time)
        it = self.its[ii]

        rls = self.sim.expand_rl(self.rls, it=it)

        grid_func = self.var.get_data(region=self.region,
                                      it=it,
                                      exclude_ghosts=self.exclude_ghosts,
                                      **self.PPkwargs)
        coords = grid_func.coords

        if not self.code_units:
            data = {rl: grid_func.scaled(rl) for rl in grid_func}
            coords = {rl: {ax: cc*Units['Length']
                           for ax, cc in coords[rl].items()}
                      for rl in grid_func}
        else:
            data = {rl: grid_func[rl] for rl in grid_func}

        if callable(self.func):
            if isinstance(self.func, np.ufunc):
                data = {rl: self.func(dd) for rl, dd in data.items()}
            elif len(signature(self.func).parameters) == 1:
                data = {rl: self.func(dd) for rl, dd in data.items()}
            else:
                data = {rl: self.func(dd, **coords[rl])
                        for rl, dd in data.items()}
        if len(self.region) == 1:
            dat = data[rls[-1]]
            xx = coords[rls[-1]][self.region]
            for rl in rls[-2::-1]:
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
    """
    TSLineAniFunc creates an animation of a vertical line that moves along the
    x-axis of a given matplotlib axes for the use with notable2.Animations.Animation.

    Paramters:
     - sim: an instance of the "Simulation" class, which is used to provide the
       current time of the simulation
     - ax: a matplotlib Axes instance that the animation will be plotted on
     - code_units: a boolean value indicating whether the time should be in code
       units or not. Default value is False.
     - **kwargs: any additional keyword arguments that will be passed to the
         matplotlib axvline function when initializing the animation
    """

    def __init__(self,
                 sim: "Simulation",
                 ax: plt.Axes,
                 code_units: bool = False,
                 **kwargs):
        self.sim = sim
        self.ax = ax
        self.kwargs = kwargs
        self.code_units = code_units

    def _get_times(self, *_, **kwargs):
        return np.array([], dtype=float)

    def _init(self):
        self.im = self.ax.axvline(0, **self.kwargs)

    def __call__(self, time: np.float_):
        if not self.code_units:
            time *= Units['Time']
        self.im.set_xdata(time)


class ContourAniFunc(GDAniFunc):
    """
    Creates an animation of a contour plot

    For the use with notable2.Animations.Animation.
    """
    image: Plot2D

    def __init__(self, *args, **kwargs):
        super().__init__(*args, contour=True, **kwargs)

        # delete cmap if colors is given
        if 'cmap' in self.kwargs and 'colors' in self.kwargs:
            del self.kwargs['cmap']

    def _init(self):
        # delete old contours if animation has looped
        if hasattr(self, 'image'):
            for rl in self.image:
                for cont in self.image[rl].collections:
                    cont.remove()

        super()._init()

        # set levels in stone
        if isinstance(self.kwargs['levels'], (int, np.integer)):
            self.kwargs['levels'] = np.linspace(self.image.norm.vmin,
                                                self.image.norm.vmax,
                                                self.kwargs['levels'])

    def __call__(self, time: np.float_):
        # this is the same as GDAniFunc.__call__ except for the setting of the
        # new data
        if time not in self.times:
            return
        ii = self.times.searchsorted(time)
        it = self.its[ii]

        rls = self.sim.expand_rl(self.rls, it=it)

        grid_func = self.var.get_data(region=self.region,
                                      it=it,
                                      exclude_ghosts=self.exclude_ghosts,
                                      **self.PPkwargs)
        coords = grid_func.coords

        if not self.code_units:
            data = {rl: grid_func.scaled(rl) for rl in grid_func}
            coords = {rl: {ax: cc*Units['Length']
                           for ax, cc in coords[rl].items()}
                      for rl in grid_func}
        else:
            data = grid_func

        if callable(self.func):
            if isinstance(self.func, np.ufunc):
                data = {rl: self.func(dd) for rl, dd in data.items()}
            elif len(signature(self.func).parameters) == 1:
                data = {rl: self.func(dd) for rl, dd in data.items()}
            else:
                data = {rl: self.func(dd, **coords[rl])
                        for rl, dd in data.items()}

        for rl in self.image:
            for cont in self.image[rl].collections:
                cont.remove()  # removes only the contours, leaves the rest intact

        kwargs = self.kwargs.copy()
        kwargs.pop('contour')
        for rl in rls:
            xx, yy = coords[rl].values()
            self.image[rl] = self.ax.contour(xx, yy, data[rl].T,
                                             norm=self.image.norm,
                                             zorder=.99+.0001*rl,
                                             **kwargs)

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


class AnyAniFunc(AniFunc):
    """
    Custom AniFunc that can be used to create any animation by passsing a
    animation function and optionally an init function

    Arguments:
        - ani_func: (callable) the animation function that will be called at
          each frame
        - init_func: (callable, None) the init function that will be called at
          the beginning of the animation if not None
    """

    def __init__(self,
                 ani_func: Callable,
                 init_func: Optional[Callable] = None,
                 ):
        self.ani_func = ani_func
        self.init_func = init_func

    def _get_times(self, *_, **kw):
        return np.array([], dtype=float)

    def _init(self):
        if self.init_func is not None:
            self.init_func()

    def __call__(self, time: np.float_):
        self.ani_func(time)
