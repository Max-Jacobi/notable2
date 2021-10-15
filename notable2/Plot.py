from inspect import signature
from typing import Optional, Sequence, Callable, Union, Any, TYPE_CHECKING
import numpy as np
import matplotlib.pyplot as plt  # type: ignore
from matplotlib.axes import Axes  # type: ignore
from matplotlib.colors import Normalize  # type: ignore
from matplotlib.colorbar import ColorbarBase  # type: ignore
from mpl_toolkits.axes_grid1 import make_axes_locatable  # type: ignore

from .Utils import Units, func_dict, Plot2D, VariableError, IterationError
from .Variable import UserVariable, UTimeSeriesVariable
if TYPE_CHECKING:
    from .Utils import Simulation, RLArgument


def handle_kwargs(var_kwargs: dict[str, Any],
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
           xlabel: Union[bool, str] = True,
           ylabel: Union[bool, str] = True,
           cbar: Optional[bool] = True,
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
        region = 'xz'
    var = sim.get_variable(key)
    if not code_units and time is not None:
        time /= Units['Time']

    its = var.available_its(region)
    if it is None and time is None:
        it = its[0]
    elif isinstance(it, int):
        if it not in its:
            raise IterationError(f"Iteration {it} for Variable {var} not in {sim}")
    elif isinstance(time, (int, float)):
        times = sim.get_time(its)
        if time > (max_time := times.max()):
            time = max_time
        if time < (min_time := times.min()):
            time = min_time
        it = its[times.searchsorted(time)][0]
    else:
        raise ValueError

    actual_rls = sim.expand_rl(rls)

    if ax is None:
        ax = plt.gca()

    var_kwargs, popped = handle_kwargs(var.kwargs, dict(func=(func, False),
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
    if isinstance(var, UserVariable):
        for kk, val in kwargs.items():
            if kk in signature(var.func).parameters:
                UVkwargs[kk] = val
    for kk in UVkwargs:
        kwargs.pop(key)

    # -------------data handling-------------------------------------------
    grid_data = var.get_data(region=region,
                             it=it,
                             exclude_ghosts=exclude_ghosts,
                             **UVkwargs)
    coords = grid_data.coords

    actual_time = grid_data.time

    if slice_ax is not None:
        ...  # TODO
    if interp_ax is not None:
        ...  # TODO

    if not code_units:
        actual_time *= Units['Time']
        data = {rl: grid_data.scaled(rl) for rl in actual_rls}
        coords = {rl: {ax: cc*Units['Length']
                       for ax, cc in coords[rl].items()}
                  for rl in actual_rls}
    else:
        data = {rl: grid_data[rl] for rl in actual_rls}

    if isinstance(func, str):
        func_str, actual_func = func_dict[func]
    elif callable(func):
        actual_func = func
        func_str = "{}"

    if callable(actual_func):
        if isinstance(actual_func, np.ufunc):
            data = {rl: actual_func(dd) for rl, dd in data.items()}
        elif len(signature(actual_func).parameters) == 1:
            data = {rl: actual_func(dd, **UVkwargs) for rl, dd in data.items()}
        else:
            coords, data = {rl: actual_func(dd, **coords[rl]) for rl, dd in data.items()}

    # -------------1D Plots------------------------------------------------
    if len(region) == 1:
        # ----------------Tidy up kwargs-----------------------------------
        for kw in ['cmap']:
            kwargs.pop(kw)

        # ----------------Plotting-----------------------------------------

        dat = data[actual_rls[-1]]
        xx, = coords[actual_rls[-1]][region]
        for rl in actual_rls[-2::-1]:
            dat_rl = data[rl]
            x_rl, = coords[rl][region]
            mask = (x_rl < xx.min()) | (x_rl > xx.max())
            xx = np.concatenate([xx, x_rl[mask]])
            dat = np.concatenate([dat, dat_rl[mask]])
        isort = np.argsort(xx)
        xx = xx[isort]
        dat = dat[isort]

        li, = ax.plot(xx, dat, **kwargs)

        # ----------------Plot Finish--------------------------------------

        if bounds is not None:
            if isinstance(bounds, (float, int)):
                bounds = (0, bounds)
            ax.xlim(*bounds)

        if label is True:
            if code_units:
                label = f"{sim.sim_name}; $t$ = {actual_time: .2f} $M_\\odot$"
            else:
                label = f"{sim.sim_name}; $t$ = {actual_time: .2f} s"
        if isinstance(label, str):
            label = label.replace('TIME}', f'{actual_time:.2f}')
            label = label.replace('IT', f'{it}')
            label = label.replace('PLOTNAME', var.plot_name.name)
            label = label.replace('SIM', sim.sim_name)
        li.set_label(label)

        if xlabel:
            if code_units:
                ax.set_xlabel(f"${region}$ " + r"[$M_\odot$]")
            else:
                ax.set_xlabel(f"${region}$ [km]")
        if ylabel:
            ylabel = func_str.format(var.plot_name.print(code_units=code_units))

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

            im[rl] = ax.imshow(dat.T, origin='lower', extent=extent, norm=norm, zorder=.9+.001*rl, **kwargs)
        plot_2d = Plot2D(im, norm, **kwargs)

        # ----------------Plot Finish--------------------------------------

        if bounds is not None:
            if isinstance(bounds, (float, int)):
                bounds = (0, bounds, 0, bounds)
            elif len(bounds) == 2:
                bounds = list(bounds)*2
            ax.axis(bounds)

        ax.set_aspect(1)

        if label is True:
            label = func_str.format(var.plot_name.print(code_units=code_units))
        if label:
            if code_units:
                ax.set_title(f"{label}\n $t$ = {actual_time:.2f} $M_\\odot$")
            else:
                ax.set_title(f"{label}\n $t$ = {actual_time:.2f} ms")

        if xlabel:
            if code_units:
                ax.set_xlabel(f"${region[0]}$ " + r"[$M_\odot$]")
            else:
                ax.set_xlabel(f"${region[0]}$ [km]")
        if ylabel:
            if code_units:
                ax.set_ylabel(f"${region[1]}$ " + r"[$M_\odot$]")
            else:
                ax.set_ylabel(f"${region[1]}$ [km]")

        if cbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            ColorbarBase(ax=cax, cmap=plot_2d.cmap, norm=plot_2d.norm)
            plt.sca(ax)

        return plot_2d


def plotTS(sim: "Simulation",
           key: str,
           # -----------Plot kwargs-------------------------------
           it_range: Optional[Sequence[int]] = None,
           time_range: Optional[Sequence[float]] = None,
           code_units: bool = False,
           ax: Optional[Axes] = None,
           every: int = 1,
           label: Union[bool, str] = True,
           xlabel: Union[bool, str] = True,
           ylabel: Union[bool, str] = True,
           # -----------Variable kwargs----------------------------
           func: Optional[Union[Callable, str, bool]] = None,
           # ------------------------------------------------------
           **kwargs):

    # -------------arguments checking and expanding------------------------
    var = sim.get_variable(key)

    its = var.available_its()
    if it_range is not None:
        its = its[(its <= it_range[1]) & (its >= it_range[0])]
    elif time_range is not None:
        times = sim.get_time(its)
        its = its[(times <= time_range[1]) & (times >= time_range[0])]

    if ax is None:
        ax = plt.gca()

    var_kwargs, popped = handle_kwargs(var.kwargs, dict(func=(func, False)))
    func = popped["func"]

    kwargs = {**var_kwargs, **kwargs}

    UVkwargs = {}
    # if hasattr(var, 'func'):
    if isinstance(var, UserVariable):
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
    ts_data = var.get_data(its=its, **UVkwargs)

    times = ts_data.times
    data = ts_data.data

    mask = np.ones_like(times).astype(bool)
    if time_range is not None:
        mask = mask & (times >= time_range[0]) & (times <= time_range[1])
    if it_range is not None:
        mask = mask & (its >= it_range[0]) & (its <= it_range[1])
    times = times[mask]
    its = its[mask]
    data = data[mask]

    if every is not None:
        times = times[::every]
        data = data[::every]

    if not code_units:
        times *= Units['Time']
        data *= var.scale_factor

    if isinstance(func, str):
        func_str, func = func_dict[func]
    else:
        func_str = "{}"

    if callable(func):
        if isinstance(func, np.ufunc):
            data = {rl: func(dd, **UVkwargs) for rl, dd in data.items()}
        elif len(signature(func).parameters) == 1:
            data = {rl: func(dd, **UVkwargs) for rl, dd in data.items()}
        else:
            times, data = {rl: func(dd, times, **UVkwargs) for rl, dd in data.items()}

    # ----------------Plotting---------------------------------------------

    li, = ax.plot(times, data, **kwargs)

    # ----------------Plot Finish------------------------------------------

    if label is True:
        label = f"SIM"

    if isinstance(label, str):
        label = label.replace('PLOTNAME', var.plot_name.name)
        label = label.replace('SIM', sim.sim_name)
    li.set_label(label)

    if xlabel:
        if code_units:
            ax.set_xlabel("$t$ " + r"[$M_\odot$]")
        else:
            ax.set_xlabel("$t$ [ms]")

    if ylabel:
        ylabel = func_str.format(var.plot_name.print(code_units=code_units))
        ax.set_ylabel(ylabel)

    return li
