from inspect import signature
from typing import Optional, Sequence, Callable, Union, Any, TYPE_CHECKING
import numpy as np
import matplotlib.pyplot as plt  # type: ignore
from matplotlib.animation import FuncAnimation  # type: ignore
from matplotlib.axes import Axes  # type: ignore
from matplotlib.colors import Normalize  # type: ignore
from matplotlib.colorbar import ColorbarBase  # type: ignore
from mpl_toolkits.axes_grid1 import make_axes_locatable  # type: ignore

from .Utils import Units, func_dict, Plot2D, IterationError, BackupException, VariableError
from .Variable import PostProcVariable, UTimeSeriesVariable
if TYPE_CHECKING:
    from .Utils import Simulation, RLArgument


def _handle_kwargs(var_kwargs: dict[str, Any],
                   pop: dict[str, tuple]
                   ) -> tuple[dict[str, Any], dict[str, Any]]:
    popped = {}
    for key, (item, default) in pop.items():
        popped[key] = default
        if key in var_kwargs:
            kw = var_kwargs.pop(key)
            popped[key] = kw
        if item is not None:
            popped[key] = item

    return var_kwargs, popped


def plotGD(sim: "Simulation",
           key: str,
           # -----------Plot kwargs-------------------------------
           it: Optional[int] = None,
           time: Optional[Union[float, int]] = None,
           region: Optional[str] = None,
           rls: "RLArgument" = None,
           code_units: bool = False,
           bounds: Optional[Sequence[float]] = None,
           exclude_ghosts: int = 0,
           ax: Optional[Axes] = None,
           mirror: Optional[str] = None,
           label: Union[bool, str] = True,
           title: Union[bool, str] = True,
           xlabel: Union[bool, str] = True,
           ylabel: Union[bool, str] = True,
           cbar: bool = True,
           contour: bool = False,
           # -----------Variable kwargs----------------------------
           func: Optional[Union[Callable, str, bool]] = None,
           slice_ax: Optional[dict[str, float]] = None,
           interp_ax: Optional[dict[str, float]] = None,
           vmax: Optional[float] = None,
           vmin: Optional[float] = None,
           symetric_around: Optional[Union[float, bool]] = None,
           norm: Optional[Normalize] = None,
           # ------------------------------------------------------
           **kwargs):

    # -------------arguments checking and expanding------------------------
    if region is None:
        region = 'xz' if sim.is_cartoon else 'xy'
    if not code_units and time is not None:
        time /= Units['Time']

    var = sim.get_variable(key)
    its = var.available_its(region)

    if it is None and time is None:
        it = its[0]
    elif isinstance(it, (int, np.integer)):
        if it not in its:
            raise IterationError(f"Iteration {it} for Variable {var} not in {sim}")
    elif isinstance(time, (int, float, np.number)):
        times = sim.get_time(its)
        if time > (max_time := times.max()):
            time = max_time
        if time < (min_time := times.min()):
            time = min_time
        it = its[times.searchsorted(time)]
    else:
        raise ValueError

    actual_rls = sim.expand_rl(rls)

    if ax is None:
        ax = plt.gca()

    var_kwargs, popped = _handle_kwargs(var.kwargs, dict(func=(func, None),
                                                         slice_ax=(slice_ax, None),
                                                         interp_ax=(interp_ax, None),
                                                         vmax=(vmax, None),
                                                         vmin=(vmin, None),
                                                         symetric_around=(symetric_around, None),
                                                         norm=(norm, Normalize(vmin, vmax))))
    func = popped["func"]
    slice_ax = popped["slice_ax"]
    interp_ax = popped["interp_ax"]
    vmax = popped["vmax"]
    vmin = popped["vmin"]
    symetric_around = popped["symetric_around"]
    norm = popped["norm"]

    kwargs = {**var_kwargs, **kwargs}

    UVkwargs = {}
    if isinstance(var, PostProcVariable):
        for kk, val in kwargs.items():
            if kk in signature(var.func).parameters:
                UVkwargs[kk] = val
    for kk in UVkwargs:
        kwargs.pop(key)

    # -------------data handling-------------------------------------------
    grid_func = var.get_data(region=region,
                             it=it,
                             exclude_ghosts=exclude_ghosts,
                             **UVkwargs)
    coords = grid_func.coords

    actual_time = grid_func.time

    if slice_ax is not None:
        ...  # TODO
    if interp_ax is not None:
        ...  # TODO

    if not code_units:
        actual_time *= Units['Time']
        data = {rl: grid_func.scaled(rl) for rl in actual_rls}
        coords = {rl: {ax: cc*Units['Length']
                       for ax, cc in coords[rl].items()}
                  for rl in actual_rls}
    else:
        data = {rl: grid_func[rl] for rl in actual_rls}

    if isinstance(func, str):
        func_str, func = func_dict[func]
    else:
        func_str = "{}"

    if callable(func):
        if isinstance(func, np.ufunc):
            data = {rl: func(dd) for rl, dd in data.items()}
        elif len(signature(func).parameters) == 1:
            data = {rl: func(dd, **UVkwargs) for rl, dd in data.items()}
        else:
            coords, data = {rl: func(dd, **coords[rl]) for rl, dd in data.items()}

    # -------------1D Plots------------------------------------------------
    if len(region) == 1:
        # ----------------Tidy up kwargs-----------------------------------
        for kw in ['cmap']:
            kwargs.pop(kw)

        # ----------------Plotting-----------------------------------------

        dat = data[actual_rls[-1]]
        xx = coords[actual_rls[-1]][region]
        for rl in actual_rls[-2::-1]:
            dat_rl = data[rl]
            x_rl = coords[rl][region]
            mask = (x_rl < xx.min()) | (x_rl > xx.max())
            xx = np.concatenate([xx, x_rl[mask]])
            dat = np.concatenate([dat, dat_rl[mask]])
        isort = np.argsort(xx)
        xx = xx[isort]
        dat = dat[isort]

        li, = ax.plot(xx, dat, **kwargs)

        # ----------------Plot Finish--------------------------------------

        if bounds is not None:
            if isinstance(bounds, (float, int, np.number)):
                bounds = (0, bounds)
            ax.set_xlim(*bounds)

        if code_units:
            t_str = f"{actual_time: .2f} $M_\\odot$"
        else:
            t_str = f"{actual_time: .2f} ms"
        if label is True:
            label = f"{sim.nice_name}; $t$ = {t_str}"
        if isinstance(label, str):
            label = label.replace('TIME', t_str)
            label = label.replace('IT', f'{it}')
            label = label.replace('PLOTNAME', func_str.format(var.plot_name.print(code_units=code_units)))
            label = label.replace('SIM', sim.nice_name)
            li.set_label(label)
        if title is True:
            title = func_str.format(var.plot_name.print(code_units=code_units))
            title = f"{title}\n $t$ = {t_str}"
        if isinstance(title, str):
            title = title.replace('TIME', t_str)
            title = title.replace('IT', f'{it}')
            title = title.replace('PLOTNAME', func_str.format(var.plot_name.print(code_units=code_units)))
            title = title.replace('SIM', sim.nice_name)
            ax.set_title(title)

        if xlabel is True:
            if code_units:
                xlabel = f"${region}$ " + r"[$M_\odot$]"
            else:
                xlabel = f"${region}$ [km]"
        if isinstance(xlabel, str):
            ax.set_xlabel(xlabel)

        if ylabel is True:
            ylabel = func_str.format(var.plot_name.print(code_units=code_units))
        if isinstance(ylabel, str):
            ax.set_ylabel(ylabel)

        return li

    # -------------2D Plots------------------------------------------------
    if len(region) == 2:
        # ----------------Norm handling------------------------------------
        if norm is None:
            norm = Normalize(vmax, vmin)
        if norm.vmax is None:
            norm.vmax = max(dat[np.isfinite(dat)].max() for dat in data.values())
        if norm.vmin is None:
            norm.vmin = min(dat[np.isfinite(dat)].min() for dat in data.values())
        if symetric_around is not None:
            dv = max(np.abs(norm.vmin - symetric_around), np.abs(norm.vmax - symetric_around))
            norm.vmin = symetric_around - dv
            norm.vmax = symetric_around + dv

        if contour:
            if 'colors' in kwargs:
                kwargs['cmap'] = None
                cbar = False
            levels = kwargs.pop('levels') if 'levels' in kwargs else 10
            if isinstance(levels, (int, np.integer)):
                levels = np.linspace(norm.vmin, norm.vmax, levels)
        # ----------------Plotting-----------------------------------------
        im = {}
        for rl in actual_rls[::-1]:
            dat = data[rl]
            xx, yy = [coords[rl][ax] for ax in region]
            dx = xx[1] - xx[0]
            dy = yy[1] - yy[0]
            extent = [xx[0]-dx/2, xx[-1]+dx/2, yy[0]-dy/2, yy[-1]+dy/2]
            if mirror is not None:
                if 'x' in mirror:
                    extent[:2] = [-xx[-1]-dx/2, -xx[0]+dx/2]
                    dat[:] = dat[::-1]
                if 'y' in mirror:
                    extent[2:] = [-yy[-1]-dy/2, -yy[0]+dy/2]
                    dat[:] = dat[:, ::-1]

            if contour:
                im[rl] = ax.contour(xx, yy, dat.T, norm=norm, levels=levels, zorder=.99+.0001*rl, **kwargs)
            else:
                im[rl] = ax.imshow(dat.T, origin='lower', extent=extent, norm=norm, zorder=.9+.001*rl, **kwargs)
        plot_2d = Plot2D(im, norm, **kwargs)

        # ----------------Plot Finish--------------------------------------

        if bounds is not None:
            if isinstance(bounds, (float, int, np.number)):
                if sim.is_cartoon:
                    bounds = (0, bounds, 0, bounds)
                else:
                    bounds = (-bounds, bounds, -bounds, bounds)
            elif len(bounds) == 2:
                bounds = list(bounds)*2
            ax.axis(bounds)

        ax.set_aspect(1)

        if code_units:
            t_str = f"{actual_time: .2f} $M_\\odot$"
            if xlabel is True:
                xlabel = f"${region[0]}$ " + r"[$M_\odot$]"
            if ylabel is True:
                ylabel = f"${region[1]}$ " + r"[$M_\odot$]"
        else:
            t_str = f"{actual_time: .2f} ms"
            if xlabel is True:
                xlabel = f"${region[0]}$ [km]"
            if ylabel is True:
                ylabel = f"${region[1]}$ [km]"

        if label is True:
            label = func_str.format(var.plot_name.print(code_units=code_units))
            label = f"{label}\n $t$ = {t_str}"
        if isinstance(label, str):
            label = label.replace('TIME', t_str)
            label = label.replace('IT', f'{it}')
            label = label.replace('PLOTNAME', func_str.format(var.plot_name.print(code_units=code_units)))
            label = label.replace('SIM', sim.nice_name)
            ax.set_label(label)

        if title is True:
            title = func_str.format(var.plot_name.print(code_units=code_units))
            title = f"{title}\n $t$ = {t_str}"
        if isinstance(title, str):
            title = title.replace('TIME', t_str)
            title = title.replace('IT', f'{it}')
            title = title.replace('PLOTNAME', func_str.format(var.plot_name.print(code_units=code_units)))
            title = title.replace('SIM', sim.nice_name)
            ax.set_title(title)

        if isinstance(xlabel, str):
            ax.set_xlabel(xlabel)

        if isinstance(ylabel, str):
            ax.set_ylabel(ylabel)

        if cbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            ColorbarBase(ax=cax, cmap=plot_2d.cmap, norm=plot_2d.norm)
            plt.sca(ax)

        return plot_2d


def plotTS(sim: "Simulation",
           key: str,
           # -----------Plot kwargs-------------------------------
           min_it: Optional[int] = None,
           max_it: Optional[int] = None,
           min_time: Optional[float] = None,
           max_time: Optional[float] = None,
           code_units: bool = False,
           ax: Optional[Axes] = None,
           every: int = 1,
           label: Union[bool, str] = True,
           xlabel: Union[bool, str] = True,
           ylabel: Union[bool, str] = True,
           # -----------Variable kwargs----------------------------
           func: Optional[Union[Callable, str, bool]] = None,
           # ------------------------------------------------------
           ** kwargs):

    # -------------arguments checking and expanding------------------------
    var = sim.get_variable(key)

    if ax is None:
        ax = plt.gca()

    var_kwargs, popped = _handle_kwargs(var.kwargs, dict(func=(func, False)))
    func = popped["func"]

    kwargs = {**var_kwargs, **kwargs}

    UVkwargs = {}
    # if hasattr(var, 'func'):
    if isinstance(var, PostProcVariable):
        for kk, val in kwargs.items():
            if kk in signature(var.func).parameters:
                UVkwargs[kk] = val
    if isinstance(var, UTimeSeriesVariable) and var.reduction is not None:
        for kk, val in kwargs.items():
            if kk in signature(var.reduction).parameters:
                UVkwargs[kk] = val

    for kk in UVkwargs:
        kwargs.pop(kk)

    # -------------data handling-------------------------------------------
    av_its = var.available_its()
    if every is not None:
        its = av_its[::every]
    if min_it is not None:
        its = its[its >= min_it]
    if max_it is not None:
        its = its[its <= max_it]
    ts_data = var.get_data(it=its, **UVkwargs)

    its = ts_data.its
    times = ts_data.times

    if code_units:
        data = ts_data.data
    else:
        data = ts_data.scaled_data

    if not code_units:
        times *= Units['Time']

    mask = np.ones_like(times).astype(bool)
    if min_time is not None:
        mask = mask & (times >= min_time)
    if max_time is not None:
        mask = mask & (times <= max_time)
    times = times[mask]
    its = its[mask]
    data = data[mask]

    if isinstance(func, str):
        func_str, func = func_dict[func]
    else:
        func_str = "{}"

    if callable(func):
        if isinstance(func, np.ufunc) or len(signature(func).parameters) == 1:
            data = func(data, **UVkwargs)
        else:
            data = func(data, times, **UVkwargs)

    # ----------------Plotting---------------------------------------------

    li, = ax.plot(times, data, **kwargs)

    # ----------------Plot Finish------------------------------------------

    if xlabel is True:
        if code_units:
            xlabel = "time [$M_\\odot$]"
        else:
            xlabel = "time [s]"
    if isinstance(xlabel, str):
        ax.set_xlabel(xlabel)

    if ylabel is True:
        ylabel = func_str.format(var.plot_name.print(code_units=code_units))
    if isinstance(ylabel, str):
        ax.set_ylabel(ylabel)

    if label is True:
        label = "SIM"
    if isinstance(label, str):
        label = label.replace('PLOTNAME', func_str.format(var.plot_name.print(code_units=code_units)))
        label = label.replace('SIM', sim.nice_name)
        li.set_label(label)

    return li


def animateGD(sim: "Simulation",
              key: str,
              # -----------Plot kwargs-------------------------------
              rls: "RLArgument" = None,
              region: Optional[str] = None,
              min_it: Optional[int] = None,
              max_it: Optional[int] = None,
              min_time: Optional[float] = None,
              max_time: Optional[float] = None,
              every: int = 1,
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

    # -------------arguments checking and expanding------------------------
    if region is None:
        region = 'xz' if sim.is_cartoon else 'xy'

    var = sim.get_variable(key)

    var_kwargs, popped = _handle_kwargs(var.kwargs, dict(func=(func, None),
                                                         slice_ax=(slice_ax, None),
                                                         interp_ax=(interp_ax, None)))
    func = popped["func"]
    slice_ax = popped["slice_ax"]
    interp_ax = popped["interp_ax"]

    kwargs = {**var_kwargs, **kwargs}

    UVkwargs = {}
    if isinstance(var, PostProcVariable):
        for kk, val in kwargs.items():
            if kk in signature(var.func).parameters:
                UVkwargs[kk] = val
    for kk in UVkwargs:
        kwargs.pop(key)

    actual_rls = sim.expand_rl(rls)

    its = var.available_its(region)
    times = sim.get_time(its)

    mask = np.ones_like(its, dtype=bool)
    if min_it is not None:
        mask = mask & (its >= min_it)
    if max_it is not None:
        mask = mask & (its <= max_it)
    if min_time is not None:
        mask = mask & (times >= min_time/(Units['Time'] if not code_units else 1))
    if max_time is not None:
        mask = mask & (times <= max_time/(Units['Time'] if not code_units else 1))
    its = its[mask][::every]
    times = times[mask][::every]

    if isinstance(func, str):
        func_str, func = func_dict[func]
    else:
        func_str = "{}"

    if title is True:
        title = func_str.format(var.plot_name.print(code_units=code_units))
        title = f"{title}\n $t$ = TIME"
    if isinstance(title, str):
        title = title.replace('PLOTNAME', func_str.format(var.plot_name.print(code_units=code_units)))
        title = title.replace('SIM', sim.nice_name)
    if label is True:
        label = sim.nice_name

    init_it = int(its[int((len(its)-1)*setup_at)])

    if xlabel is True:
        if code_units:
            xlabel = f"${region[0]}$ " + r"[$M_\odot$]"
        else:
            xlabel = f"${region[0]}$ [km]"
    if ylabel is True:
        if len(region) == 1:
            ylabel = func_str.format(var.plot_name.print(code_units=code_units))
        else:
            if code_units:
                ylabel = f"${region[1]}$ " + r"[$M_\odot$]"
            else:
                ylabel = f"${region[1]}$ [km]"

    # ----------------setup plot object------------------------------------------

    image = sim.plotGD(key=key,
                       it=init_it,
                       region=region,
                       rls=actual_rls,
                       code_units=code_units,
                       title=title,
                       label=label,
                       func=func,
                       slice_ax=slice_ax,
                       interp_ax=interp_ax,
                       exclude_ghosts=exclude_ghosts,
                       **kwargs)
    ax = image.axes
    if label:
        ax.legend()

    # ----------------get animation function------------------------------------------
    def _animate(ii):
        it = its[ii]
        time = times[ii]

        grid_func = var.get_data(region=region,
                                 it=it,
                                 exclude_ghosts=exclude_ghosts,
                                 **UVkwargs)
        coords = grid_func.coords
        if not code_units:
            data = {rl: grid_func.scaled(rl) for rl in actual_rls}
            coords = {rl: {ax: cc*Units['Length']
                           for ax, cc in coords[rl].items()}
                      for rl in actual_rls}
        else:
            data = {rl: grid_func[rl] for rl in actual_rls}

        if callable(func):
            if isinstance(func, np.ufunc):
                data = {rl: func(dd) for rl, dd in data.items()}
            elif len(signature(func).parameters) == 1:
                data = {rl: func(dd, **UVkwargs) for rl, dd in data.items()}
            else:
                coords, data = {rl: func(dd, **coords[rl]) for rl, dd in data.items()}
        if len(region) == 1:
            dat = data[actual_rls[-1]]
            xx = coords[actual_rls[-1]][region]
            for rl in actual_rls[-2::-1]:
                dat_rl = data[rl]
                x_rl = coords[rl][region]
                mask = (x_rl < xx.min()) | (x_rl > xx.max())
                xx = np.concatenate([xx, x_rl[mask]])
                dat = np.concatenate([dat, dat_rl[mask]])
            isort = np.argsort(xx)
            xx = xx[isort]
            dat = dat[isort]

            image.set_data(xx, dat)
        elif len(region) == 2:
            for rl in actual_rls[::-1]:
                im = image[rl]
                xx, yy = [coords[rl][ax] for ax in region]
                dx = xx[1] - xx[0]
                dy = yy[1] - yy[0]
                extent = [xx[0]-dx/2, xx[-1]+dx/2, yy[0]-dy/2, yy[-1]+dy/2]
                im.set_extent(extent)
                im.set_data(data[rl].T)
            ax.set_aspect(1)

        if code_units:
            t_str = f"{time: .2f} $M_\\odot$"
        else:
            t_str = f"{time*Units['Time']: .2f} ms"
        new_title = title
        if isinstance(new_title, str):
            new_title = new_title.replace('TIME', t_str)
            new_title = new_title.replace('IT', f'{it}')
            ax.set_title(new_title)

    ani = FuncAnimation(ax.figure, _animate, frames=len(its))
    plt.close(ax.figure)
    return ani
