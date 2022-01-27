from inspect import signature
from typing import Optional, Sequence, Callable, Union, Any, TYPE_CHECKING
from collections.abc import Iterable
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt  # type: ignore
from matplotlib.animation import FuncAnimation  # type: ignore
from matplotlib.axes import Axes  # type: ignore
from matplotlib.colors import Normalize, LogNorm  # type: ignore
from matplotlib.colorbar import ColorbarBase  # type: ignore
from mpl_toolkits.axes_grid1 import make_axes_locatable  # type: ignore

from .Utils import Units, func_dict, Plot2D, IterationError, BackupException, VariableError
from .Variable import PostProcVariable, PPTimeSeriesVariable
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


def _handle_PPkwargs(kwargs, var):
    PPkwargs = {}
    if isinstance(var, PostProcVariable):
        for kk, val in kwargs.items():
            if kk in var.PPkeys:
                PPkwargs[kk] = val
    for kk in PPkwargs:
        kwargs.pop(kk)
    if isinstance(var, PostProcVariable):
        for dvar in var.dependencies:
            kwargs, new_PPkwargs = _handle_PPkwargs(kwargs, dvar)
            PPkwargs = {**PPkwargs, **new_PPkwargs}
    return kwargs, PPkwargs


def plotGD(sim: "Simulation",
           key: str,
           # -----------Plot kwargs-------------------------------
           it: Optional[int] = None,
           time: Optional[Union[float, int]] = None,
           region: Optional[str] = None,
           rls: "RLArgument" = None,
           code_units: bool = False,
           bounds: Optional[Union[float, Sequence[float]]] = None,
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
        if sim.t_merg is not None:
            times -= sim.t_merg
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

    kwargs, PPkwargs = _handle_PPkwargs(kwargs, var)

    # -------------data handling-------------------------------------------
    grid_func = var.get_data(region=region,
                             it=it,
                             exclude_ghosts=exclude_ghosts,
                             **PPkwargs)
    coords = grid_func.coords

    actual_time = grid_func.time
    if sim.t_merg is not None:
        actual_time -= sim.t_merg

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
            data = {rl: func(dd) for rl, dd in data.items()}
        else:
            coords, data = {rl: func(dd, **coords[rl]) for rl, dd in data.items()}

    if sim.t_merg is not None:
        t_str = r"$t - t_{\rm merg}$ = "
    else:
        t_str = "$t$ = "

    if code_units:
        t_str += f"{actual_time: .2f} $M_\\odot$"
        x_str = f"${region}$ " + r"[$M_\odot$]"
        if xlabel is True:
            xlabel = f"${region[0]}$ " + r"[$M_\odot$]"
    else:
        t_str += f"{actual_time: .2f} ms"
        if xlabel is True:
            xlabel = f"${region[0]}$ [km]"

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
                bounds = float(bounds)
                if sim.is_cartoon:
                    ax.set_xlim(0, bounds)
                else:
                    ax.set_xlim(-bounds, bounds)
            elif isinstance(bounds, Iterable):
                ax.set_xlim(*bounds)
        ax.set_ylim(vmin, vmax)

        if label is True:
            label = f"{sim.nice_name}; {t_str}"
        if isinstance(label, str):
            label = label.replace('TIME', t_str)
            label = label.replace('IT', f'{it}')
            label = label.replace('PLOTNAME', func_str.format(var.plot_name.print(code_units=code_units, **PPkwargs)))
            label = label.replace('SIM', sim.nice_name)
            li.set_label(label)
        if title is True:
            title = func_str.format(var.plot_name.print(code_units=code_units, **PPkwargs))
            title = f"{title}\n{t_str}"
        if isinstance(title, str):
            title = title.replace('TIME', t_str)
            title = title.replace('IT', f'{it}')
            title = title.replace('PLOTNAME', func_str.format(var.plot_name.print(code_units=code_units, **PPkwargs)))
            title = title.replace('SIM', sim.nice_name)
            ax.set_title(title)

        if isinstance(xlabel, str):
            ax.set_xlabel(xlabel)

        if ylabel is True:
            ylabel = func_str.format(var.plot_name.print(code_units=code_units, **PPkwargs))
        if isinstance(ylabel, str):
            ax.set_ylabel(ylabel)

        return li

    # -------------2D Plots------------------------------------------------
    if len(region) == 2:
        # ----------------Norm handling------------------------------------
        if norm is None:
            norm = Normalize(vmax, vmin)
        if norm.vmax is None:
            norm.vmax = max((dat[nan_mask].max() if np.any(nan_mask := np.isfinite(dat)) else 1) for dat in data.values())
        if norm.vmin is None:
            norm.vmin = min((dat[nan_mask].min() if np.any(nan_mask := np.isfinite(dat)) else 0) for dat in data.values())
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
                bounds = float(bounds)
                if sim.is_cartoon:
                    bounds = (0., bounds, 0., bounds)
                else:
                    bounds = (-bounds, bounds, -bounds, bounds)
            elif len(bounds) == 2:
                bounds = list(bounds)*2
            ax.axis(bounds)

        # ax.set_aspect(1)

        if label is True:
            label = func_str.format(var.plot_name.print(code_units=code_units, **PPkwargs))
            label = f"{label}\n{t_str}"
        if isinstance(label, str):
            label = label.replace('TIME', t_str)
            label = label.replace('IT', f'{it}')
            label = label.replace('PLOTNAME', func_str.format(var.plot_name.print(code_units=code_units, **PPkwargs)))
            label = label.replace('SIM', sim.nice_name)
            ax.set_label(label)

        if title is True:
            title = func_str.format(var.plot_name.print(code_units=code_units, **PPkwargs))
            title = f"{title}\n{t_str}"
        if isinstance(title, str):
            title = title.replace('TIME', t_str)
            title = title.replace('IT', f'{it}')
            title = title.replace('PLOTNAME', func_str.format(var.plot_name.print(code_units=code_units, **PPkwargs)))
            title = title.replace('SIM', sim.nice_name)
            ax.set_title(title)

        if isinstance(xlabel, str):
            ax.set_xlabel(xlabel)

        if code_units:
            if ylabel is True:
                ylabel = f"${region[1]}$ " + r"[$M_\odot$]"
        else:
            if ylabel is True:
                ylabel = f"${region[1]}$ [km]"
        if isinstance(ylabel, str):
            ax.set_ylabel(ylabel)

        if cbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            ColorbarBase(ax=cax, cmap=plot_2d.cmap, norm=plot_2d.norm)
            try:
                plt.sca(ax)
            except ValueError:
                pass

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
           **kwargs):

    # -------------arguments checking and expanding------------------------
    var = sim.get_variable(key)

    if ax is None:
        ax = plt.gca()

    var_kwargs, popped = _handle_kwargs(var.kwargs, dict(func=(func, False)))
    func = popped["func"]

    kwargs = {**var_kwargs, **kwargs}

    kwargs, PPkwargs = _handle_PPkwargs(kwargs, var)
    # -------------data handling-------------------------------------------
    av_its = var.available_its()
    if every is not None:
        its = av_its[::every]
    if min_it is not None:
        its = its[its >= min_it]
    if max_it is not None:
        its = its[its <= max_it]
    ts_data = var.get_data(it=its, **PPkwargs)

    its = ts_data.its
    times = ts_data.times
    if sim.t_merg is not None:
        times -= sim.t_merg
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
            data = func(data)
        else:
            data = func(data, times)

    # ----------------Plotting---------------------------------------------

    li, = ax.plot(times, data, **kwargs)

    # ----------------Plot Finish------------------------------------------

    if xlabel is True:
        if sim.t_merg is not None:
            xlabel = r"$t - t_{\rm merg}$"
        else:
            xlabel = "$t$"
        if code_units:
            xlabel += " [$M_\\odot$]"
        else:
            xlabel += " [ms]"
    if isinstance(xlabel, str):
        ax.set_xlabel(xlabel)

    if ylabel is True:
        ylabel = func_str.format(var.plot_name.print(code_units=code_units, **PPkwargs))
    if isinstance(ylabel, str):
        ax.set_ylabel(ylabel)

    if label is True:
        label = "SIM"
    if isinstance(label, str):
        label = label.replace('PLOTNAME', func_str.format(var.plot_name.print(code_units=code_units, **PPkwargs)))
        label = label.replace('SIM', sim.nice_name)
        li.set_label(label)

    return li


def plotHist(sim: "Simulation",
             xkey: str,
             ykey: str,
             xfunc: Optional[Union[Callable, str, bool]] = None,
             yfunc: Optional[Union[Callable, str, bool]] = None,
             # -----------Plot kwargs-------------------------------
             it: Optional[int] = None,
             time: Optional[Union[float, int]] = None,
             rls: "RLArgument" = None,
             code_units: bool = False,
             ax: Optional[Axes] = None,
             title: Union[bool, str] = True,
             xlabel: Union[bool, str] = True,
             ylabel: Union[bool, str] = True,
             cmap: str = None,
             cbar: bool = True,
             norm: Optional[Normalize] = None,
             # ------------------------------------------------------
             **kwargs):
    # -------------arguments checking and expanding------------------------
    if norm is None:
        norm = LogNorm(clip=True)
    if not code_units and time is not None:
        time /= Units['Time']

    xvar = sim.get_variable(xkey)
    yvar = sim.get_variable(ykey)

    region = 'xz' if sim.is_cartoon else 'xyz'

    its = np.intersect1d(xvar.available_its(region), yvar.available_its(region))

    if it is None and time is None:
        it = its[0]
    elif isinstance(it, (int, np.integer)):
        if it not in its:
            raise IterationError(f"Iteration {it} for Variables {xvar} and {yvar} not in {sim}")
    elif isinstance(time, (int, float, np.number)):
        times = sim.get_time(its)
        if sim.t_merg is not None:
            times -= sim.t_merg
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

    _, popped = _handle_kwargs(xvar.kwargs, dict(func=(xfunc, None)))
    xfunc = popped["func"]
    _, popped = _handle_kwargs(yvar.kwargs, dict(func=(yfunc, None),
                                                 cmap=(cmap, None)))
    cmap = popped["cmap"]
    yfunc = popped["func"]

    kwargs, xPPkwargs = _handle_PPkwargs(kwargs, xvar)
    kwargs, yPPkwargs = _handle_PPkwargs(kwargs, yvar)

    # -------------data handling-------------------------------------------
    xgrid_func = xvar.get_data(region=region,
                               it=it,
                               **xPPkwargs)
    ygrid_func = yvar.get_data(region=region,
                               it=it,
                               **yPPkwargs)
    dgrid_func = sim.get_data('dens',
                              region=region,
                              it=it)
    wgrid_func = sim.get_data('reduce-weights',
                              region=region,
                              it=it)

    coords = xgrid_func.coords

    actual_time = xgrid_func.time
    if sim.t_merg is not None:
        actual_time -= sim.t_merg

    if not code_units:
        actual_time *= Units['Time']
        xdata = {rl: xgrid_func.scaled(rl) for rl in actual_rls}
        ydata = {rl: ygrid_func.scaled(rl) for rl in actual_rls}
        coords = {rl: {ax: cc*Units['Length']
                       for ax, cc in coords[rl].items()}
                  for rl in actual_rls}
    else:
        xdata = {rl: xgrid_func[rl] for rl in actual_rls}
        ydata = {rl: ygrid_func[rl] for rl in actual_rls}
    ddata = {rl: dgrid_func[rl] for rl in actual_rls}
    wdata = {rl: wgrid_func[rl] for rl in actual_rls}

    if isinstance(xfunc, str):
        xfunc_str, xfunc = func_dict[xfunc]
    else:
        xfunc_str = "{}"
    if isinstance(yfunc, str):
        yfunc_str, yfunc = func_dict[yfunc]
    else:
        yfunc_str = "{}"

    if callable(xfunc):
        if isinstance(xfunc, np.ufunc):
            xdata = {rl: xfunc(dd) for rl, dd in xdata.items()}
        elif len(signature(xfunc).parameters) == 1:
            xdata = {rl: xfunc(dd) for rl, dd in xdata.items()}
        else:
            coords, xdata = {rl: xfunc(dd, **coords[rl]) for rl, dd in xdata.items()}

    if callable(yfunc):
        if isinstance(yfunc, np.ufunc):
            ydata = {rl: yfunc(dd) for rl, dd in ydata.items()}
        elif len(signature(yfunc).parameters) == 1:
            ydata = {rl: yfunc(dd) for rl, dd in ydata.items()}
        else:
            coords, ydata = {rl: yfunc(dd, **coords[rl]) for rl, dd in ydata.items()}

    xdat = np.concatenate([xd.ravel() for xd in xdata.values()])
    ydat = np.concatenate([yd.ravel() for yd in ydata.values()])
    mdat = []
    for coord, rw, dd in zip(coords.values(),
                             wdata.values(),
                             ddata.values()):
        dx = {ax: cc[1] - cc[0] for ax, cc in coord.items()}
        coord = dict(zip(coord, np.meshgrid(*coord.values(), indexing='ij')))
        if sim.is_cartoon:
            vol = 2*np.pi*dx['x']*dx['z']*np.abs(coord['x'])
        else:
            vol = dx['x']*dx['y']*dx['z']
        mdat.append((vol*rw*dd).ravel())
    mdat = np.concatenate(mdat)

    # ----------------Plotting-----------------------------------------
    im = ax.hist2d(xdat, ydat, weights=mdat, norm=norm, cmap=cmap, **kwargs)

    if cbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        ColorbarBase(ax=cax, cmap=im[-1].get_cmap(), norm=im[-1].norm)
        cax.set_title(r'$M_\odot$')
        try:
            plt.sca(ax)
        except ValueError:
            pass

    if xlabel is True:
        xlabel = xfunc_str.format(xvar.plot_name.print(code_units=code_units, **xPPkwargs))
    if isinstance(xlabel, str):
        ax.set_xlabel(xlabel)
    if ylabel is True:
        ylabel = yfunc_str.format(yvar.plot_name.print(code_units=code_units, **yPPkwargs))
    if isinstance(ylabel, str):
        ax.set_ylabel(ylabel)

    if sim.t_merg is not None:
        t_str = r"$t - t_{\rm merg}$ = "
    else:
        t_str = "$t$ = "
    if code_units:
        t_str += f"{actual_time: .2f} $M_\\odot$"
    else:
        t_str += f"{actual_time: .2f} ms"
    if title is True:
        title = t_str
    if isinstance(title, str):
        title = title.replace('TIME', t_str)
        title = title.replace('IT', f'{it}')
        title = title.replace('SIM', sim.nice_name)
        ax.set_title(title)
    return im


def animateGD(sim: "Simulation",
              *args,
              fig: Optional[plt.Figure] = None,
              ax: Optional[plt.Axes] = None,
              min_it: Optional[int] = None,
              max_it: Optional[int] = None,
              min_time: Optional[float] = None,
              max_time: Optional[float] = None,
              every: int = 1,
              **kwargs):

    from .Animations import Animation

    if fig is None:
        if ax is None:
            fig, ax = plt.subplots(1)
        else:
            fig = ax.figure
    elif ax is None:
        ax = fig.gca()

    ani = Animation(min_time=min_time,
                    max_time=max_time,
                    every=every)
    ani.add_animation(sim.GDAniFunc(*args, ax=ax, **kwargs))

    return ani.animate(fig)
