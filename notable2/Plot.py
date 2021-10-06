
from inspect import signature
from typing import Optional, Sequence, Callable, Union, Any, TYPE_CHECKING
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import Normalize
from matplotlib.colorbar import ColorbarBase
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .Utils import Units, func_dict, Plot2D
if TYPE_CHECKING:
    from .Utils import Simulation, RLArgument, NDArray, Variable


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


def plot(sim: "Simulation",
         key: str,
         # -----------Plot kwargs-------------------------------
         it: Optional[int] = None,
         time: Optional[float] = None,
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
         # progress_bar: bool = False,
         # -----------Variable kwargs----------------------------
         func: Optional[Union[Callable, str, bool]] = None,
         slice_ax: Optional[dict[str, float]] = None,
         interp_ax: Optional[dict[str, float]] = None,
         vmax: Optional[float] = None,
         vmin: Optional[float] = None,
         symetric_around: Optional[Union[float, bool]] = None,
         norm: Optional[Normalize] = None,
         # ------------------------------------------------------
         UVkwargs: dict[str, Any] = {},
         **kwargs):

    # -------------arguments checking and expanding------------------------
    if region is None:
        region = 'xz'
    var = sim.get_variable(key)
    it = sim.lookup_it(var, region, time, it)
    rls = sim.expand_rl(rls)
    if ax is None:
        ax = plt.gca()

    var_kwargs, popped = handle_kwargs(var.kwargs, dict(func=(func, False),
                                                        slice_ax=(slice_ax, None),
                                                        interp_ax=(interp_ax, None),
                                                        vmax=(vmax, None),
                                                        vmin=(vmin, None),
                                                        symetric_around=(symetric_around, None),
                                                        norm=(norm, Normalize())))
    func = popped["func"]
    slice_ax = popped["slice_ax"]
    interp_ax = popped["interp_ax"]
    vmax = popped["vmax"]
    vmin = popped["vmin"]
    symetric_around = popped["symetric_around"]
    norm = popped["norm"]

    kwargs = {**var_kwargs, **kwargs}

    # -------------data handling-------------------------------------------
    coords = sim.get_coords(region, it, rls, exclude_ghosts=exclude_ghosts)
    data = var.get_data(region, it, exclude_ghosts=exclude_ghosts, **UVkwargs)

    time = data.time
    if not code_units:
        time *= Units['Time']

    if slice_ax is not None:
        ...  # TODO
    if interp_ax is not None:
        ...  # TODO

    if not code_units:
        data = {rl: data.scaled(rl) for rl in rls}
        coords = {rl: [cc*Units['Length']
                       for cc in coords[rl]]
                  for rl in rls}
    else:
        data = {rl: data[rl] for rl in rls}

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
            coords, data = {rl: func(coords, dd) for rl, dd in data.items()}

    # -------------1D Plots------------------------------------------------
    if len(region) == 1:
        # ----------------Tidy up kwargs-----------------------------------
        for key in ['cmap']:
            kwargs.pop(key)

        # ----------------Plotting-----------------------------------------

        dat = data[rls[-1]]
        xx, = coords[rls[-1]]
        for rl in rls[-2::-1]:
            dat_rl = data[rl]
            x_rl, = coords[rl]
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
                label = f"{sim.sim_name}; $t$ = {time: .2f} $M_\\odot$"
            else:
                label = f"{sim.sim_name}; $t$ = {time: .2f} s"
        if isinstance(label, str):
            label = label.replace('TIME}', f'{time:.2f}')
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

    # -------------2D Plots------------------------------------------------
    if len(region) == 2:
        # ----------------Norm handling------------------------------------
        if norm is None:
            norm = Normalize(vmax, vmin)
        if norm.vmax is None:
            norm.vmax = max(dat.max() for dat in data.values())
        if norm.vmin is None:
            norm.vmin = min(dat.min() for dat in data.values())
        if symetric_around is not None:
            dv = max(np.abs(norm.vmin - symetric_around), np.abs(norm.vmax - symetric_around))
            norm.vmin = symetric_around - dv
            norm.vmax = symetric_around + dv

        # ----------------Plotting-----------------------------------------
        im = {}
        for rl in rls[::-1]:
            dat = data[rl]
            xx, yy = coords[rl]
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
        im = Plot2D(im, norm, **kwargs)

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
                ax.set_title(f"{label}\n $t$ = {time:.2f} $M_\\odot$")
            else:
                ax.set_title(f"{label}\n $t$ = {time:.2f} ms")

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
            ColorbarBase(ax=cax, cmap=im.cmap, norm=im.norm)
            plt.sca(ax)
