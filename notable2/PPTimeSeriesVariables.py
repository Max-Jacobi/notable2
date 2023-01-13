import numpy as np
from scipy.integrate import cumtrapz  # type: ignore
from notable2.Reductions import integral, integral_2D, sphere_surface_integral
from notable2.Reductions import mean, minimum, maximum
from notable2.Utils import Units, RUnits


def _ye_eq(Le, La, Lne, Lna, *_, **__):
    mm = (Lne > 0) * (Lna > 0)
    Ee = (Le[mm]/Lne[mm])
    Ea = (La[mm]/Lna[mm])
    ee = Ee*1.2
    ea = Ea*1.2
    We = 1 + 1.01*Ee
    Wa = 1 - 7.22*Ea
    DD = 2.067e-6
    # from Rosswog & Korobkin 2022 (eq 1)
    ye = La[mm] * Wa * (ea - 2*DD + 1.2*DD**2/ea)
    ye /= Le[mm] * We * (ee - 2*DD + 1.2*DD**2/ee)
    res = np.zeros_like(Le)
    res[mm] = (1 + ye)**-1
    return res


def _ident(xx, *_, **__):
    return xx


def _m_decomp_r(dd, mm, x, y, *_, **__):
    phi = np.arctan2(y, x)
    return dd*np.cos(mm*phi)


def _m_decomp_i(dd, mm, x, y, *_, **__):
    phi = np.arctan2(y, x)
    return -dd*np.sin(mm*phi)


def _mass_flow(Vr, dens, *_, **kw):
    Vr[Vr < 0] = 0
    return dens*Vr


def _times_domain_volume(dependencies, its, var, func, **kwargs):
    bmass = dependencies[0].get_data(it=its)
    coords = var.sim.get_coords('xyz', 0, exclude_ghosts=3)[0]
    vol = np.prod([cc[-1] - cc[0] for cc in coords.values()])
    return func(bmass.data)*vol


def _rho_bulk(dependencies, its, var, **kwargs):
    region = 'xz' if var.sim.is_cartoon else 'xyz'
    result = np.zeros_like(its, dtype=float)

    n_its = len(its)
    for ii, it in enumerate(its):
        rl = var.sim.finest_rl[it]
        if var.sim.verbose and n_its > 1:
            print(f"{var.sim.sim_name} - {var.key}: Processing iteration {it} ({ii/n_its*100:.1f}%)",
                  end=('\r' if var.sim.verbose == 1 else '\n'))
        dens = dependencies[0].get_data(region=region, it=it)
        rho = dependencies[1].get_data(region=region, it=it)
        dds = dens[rl]
        rrs = rho[rl]
        coords = dens.coords[rl]

        vols = np.ones_like(dds)
        if var.sim.is_cartoon:
            vols *= np.abs(coords['x'])

        dds = dds.ravel()
        rrs = rrs.ravel()
        vols = vols.ravel()
        mass = vols*dds

        fin_msk = np.isfinite(dds)
        mass = mass[fin_msk]
        vols = vols[fin_msk]
        rrs = rrs[fin_msk]

        isort = np.argsort(rrs)[::-1]

        rrs = rrs[isort]
        mass = np.cumsum(mass[isort])
        vols = np.cumsum(vols[isort])
        Cs = mass/vols**.333333333

        result[ii] = rrs[Cs.argmax()]
    return result


def _time_int(dd, tt, **_):
    res = np.zeros_like(dd)
    res[1:] = cumtrapz(dd, tt)
    return res


def _nan_mask(mask):
    ret = np.ones_like(mask, dtype=float)
    ret[~mask] = np.nan
    return ret


def _mean(data, rho, rho_cont=1e13*RUnits['Rho'], rho2_cont=None, *_, **kw):
    mean = data*_nan_mask(rho < rho_cont)
    if rho2_cont is not None:
        mean *= _nan_mask(rho >= rho2_cont)
    return mean


def _rho_cont_format_func(rho_cont=1e13*RUnits['Rho'], code_units=False):
    if code_units:
        return f"{rho_cont:.0f} " + r'M_\odot^{-2}$'
    else:
        return f"{rho_cont*Units['Rho']:.0e}"+r"\,$g cm$^{-3}$"


def _radius_format_func(radius, code_units=False):
    if code_units:
        return f"{radius:.0f} " + r'M_\odot$'
    else:
        return f"{radius*Units['Length']:.0e}"+r"\,$km"


pp_variables = {
    'psi-max': dict(
        backups=["phi-max-BSNN"],
        dependencies=('phi-min',),
        func=lambda phi, *_, **kw: phi**-.5,
        plot_name_kwargs=dict(
            name="maximum conformal factor",
        ),
    ),
    'psi-min': dict(
        backups=["phi-min-BSNN"],
        dependencies=('phi-max',),
        func=lambda phi, *_, **kw: phi**-.5,
        plot_name_kwargs=dict(
            name="minimum conformal factor",
        ),
    ),
    'psi-max-BSSN': dict(
        dependencies=('phi-max-BSSN',),
        func=lambda phi, *_, **kw: np.exp(phi),
        plot_name_kwargs=dict(
            name="maximum conformal factor",
        ),
    ),
    'psi-min-BSSN': dict(
        dependencies=('phi-min-BSSN',),
        func=lambda phi, *_, **kw: np.exp(phi),
        plot_name_kwargs=dict(
            name="minimum conformal factor",
        ),
    ),
    'baryon-mass': dict(
        backups=['baryon-mass-pp'],
        dependencies=('b-mass-dummy',),
        func=lambda dd, *_, **kw: dd*2,
        plot_name_kwargs=dict(
            name="total baryon mass",
            unit=r"$M_\odot$"
        ),
        reduction=_times_domain_volume,
    ),
    'baryon-mass-pp': dict(
        dependencies=('dens',),
        func=lambda dd, *_, **kw: dd*2,
        plot_name_kwargs=dict(
            name="total baryon mass",
            unit=r"$M_\odot$"
        ),
        reduction=integral
    ),
    'L-e-norm': dict(
        backups=["L-e-pp"],
        dependencies=('L-e',),
        func=lambda dd, *_, **kw: 2*dd,
        plot_name_kwargs=dict(
            name=r"electron neutrino luminosity",
            unit=r"$10^{51}\,\mathrm{erg\,s}^{-1}$",
            code_unit=r"$10^{51} m_\odot m_\odot^{-1}$"
        ),
        reduction=_times_domain_volume,
        scale_factor=Units["Energy"]/Units["Time"]*1e3
    ),
    'L-a-norm': dict(
        backups=["L-a-pp"],
        dependencies=('L-a',),
        func=lambda dd, *_, **kw: 2*dd,
        plot_name_kwargs=dict(
            name=r"electron antineutrino luminosity",
            unit=r"$10^{51}\,\mathrm{erg\,s}^{-1}$",
            code_unit=r"$10^{51} M_\odot M_\odot^{-1}$"
        ),
        reduction=_times_domain_volume,
        scale_factor=Units["Energy"]/Units["Time"]*1e3
    ),
    'L-x-norm': dict(
        backups=["L-pp"],
        dependencies=('L-x',),
        func=lambda dd, *_, **kw: 2*dd,
        plot_name_kwargs=dict(
            name=r"x neutrino luminosity",
            unit=r"$10^{51}\,\mathrm{erg\,s}^{-1}$",
            code_unit=r"$10^{51} M_\odot M_\odot^{-1}$"
        ),
        reduction=_times_domain_volume,
        scale_factor=Units["Energy"]/Units["Time"]*1e3
    ),
    'L-e-pp': dict(
        dependencies=('L-nu-e',),
        func=lambda dd, *_, **kw: 2*dd,
        plot_name_kwargs=dict(
            name=r"electron neutrino luminosity",
            unit=r"$10^{51}\,\mathrm{erg\,s}^{-1}$",
            code_unit=r"$10^{51} m_\odot m_\odot^{-1}$"
        ),
        reduction=integral,
        scale_factor=Units["Energy"]/Units["Time"]*1e3
    ),
    'L-a-pp': dict(
        dependencies=('L-nu-a',),
        func=lambda dd, *_, **kw: 2*dd,
        plot_name_kwargs=dict(
            name=r"electron antineutrino luminosity",
            unit=r"$10^{51}\,\mathrm{erg\,s}^{-1}$",
            code_unit=r"$10^{51} M_\odot M_\odot^{-1}$"
        ),
        reduction=integral,
        scale_factor=Units["Energy"]/Units["Time"]*1e3
    ),
    'L-x-pp': dict(
        dependencies=('L-nu-x',),
        func=lambda dd, *_, **kw: 2*dd,
        plot_name_kwargs=dict(
            name=r"x neutrino luminosity",
            unit=r"$10^{51}\,\mathrm{erg\,s}^{-1}$",
            code_unit=r"$10^{51} M_\odot M_\odot^{-1}$"
        ),
        reduction=integral,
        scale_factor=Units["Energy"]/Units["Time"]*1e3
    ),
    'L-tot': dict(
        dependencies=('L-e', 'L-a', 'L-x'),
        func=lambda e, a, x, *_, **kw: e+a+x,
        plot_name_kwargs=dict(
            name=r"total neutrino luminosity",
            unit=r"$10^{51}\,\mathrm{erg\,s}^{-1}$",
            code_unit=r"$10^{51} M_\odot M_\odot^{-1}$"
        ),
        scale_factor=Units["Energy"]/Units["Time"]*1e3
    ),
    'L-e-bulk': dict(
        dependencies=('L-nu-e', 'rho', 'rho-bulk'),
        func=lambda L, dd, dns, *_, **kw: L*(dd >= dns),
        plot_name_kwargs=dict(
            name=r"bulk electron neutrino luminosity",
            unit=r"$10^{51}\,\mathrm{erg\,s}^{-1}$",
            code_unit=r"$10^{51} M_\odot M_\odot^{-1}$"
        ),
        reduction=integral,
        scale_factor=Units["Energy"]/Units["Time"]*1e3
    ),
    'L-a-bulk': dict(
        dependencies=('L-nu-a', 'rho', 'rho-bulk'),
        func=lambda L, dd, dns, *_, **kw: L*(dd >= dns),
        plot_name_kwargs=dict(
            name=r"bulk electron antineutrino luminosity",
            unit=r"$10^{51}\,\mathrm{erg\,s}^{-1}$",
            code_unit=r"$10^{51} M_\odot M_\odot^{-1}$"
        ),
        reduction=integral,
        scale_factor=Units["Energy"]/Units["Time"]*1e3
    ),
    'L-x-bulk': dict(
        dependencies=('L-nu-x', 'rho', 'rho-bulk'),
        func=lambda L, dd, dns, *_, **kw: L*(dd >= dns),
        plot_name_kwargs=dict(
            name=r"bulk x neutrino luminosity",
            unit=r"$10^{51}\,\mathrm{erg\,s}^{-1}$",
            code_unit=r"$10^{51} M_\odot M_\odot^{-1}$"
        ),
        reduction=integral,
        scale_factor=Units["Energy"]/Units["Time"]*1e3
    ),
    'L-tot-bulk': dict(
        dependencies=('L-e-bulk', 'L-a-bulk', 'L-x-bulk'),
        func=lambda e, a, x, *_, **kw: e+a+x,
        plot_name_kwargs=dict(
            name=r"total bulk neutrino luminosity",
            unit=r"$10^{51}\,\mathrm{erg\,s}^{-1}$",
            code_unit=r"$10^{51} M_\odot M_\odot^{-1}$"
        ),
        scale_factor=Units["Energy"]/Units["Time"]*1e3
    ),
    'L-e-disk': dict(
        dependencies=('L-nu-e', 'rho', 'rho-bulk'),
        func=lambda L, dd, dns, *_, **kw: L*(dd < dns),
        plot_name_kwargs=dict(
            name=r"disk electron neutrino luminosity",
            unit=r"$10^{51}\,\mathrm{erg\,s}^{-1}$",
            code_unit=r"$10^{51} M_\odot M_\odot^{-1}$"
        ),
        reduction=integral,
        scale_factor=Units["Energy"]/Units["Time"]*1e3
    ),
    'L-a-disk': dict(
        dependencies=('L-nu-a', 'rho', 'rho-bulk'),
        func=lambda L, dd, dns, *_, **kw: L*(dd < dns),
        plot_name_kwargs=dict(
            name=r"disk electron antineutrino luminosity",
            unit=r"$10^{51}\,\mathrm{erg\,s}^{-1}$",
            code_unit=r"$10^{51} M_\odot M_\odot^{-1}$"
        ),
        reduction=integral,
        scale_factor=Units["Energy"]/Units["Time"]*1e3
    ),
    'L-x-disk': dict(
        dependencies=('L-nu-x', 'rho', 'rho-bulk'),
        func=lambda L, dd, dns, *_, **kw: L*(dd < dns),
        plot_name_kwargs=dict(
            name=r"disk x neutrino luminosity",
            unit=r"$10^{51}\,\mathrm{erg\,s}^{-1}$",
            code_unit=r"$10^{51} M_\odot M_\odot^{-1}$"
        ),
        reduction=integral,
        scale_factor=Units["Energy"]/Units["Time"]*1e3
    ),
    'L-tot-disk': dict(
        dependencies=('L-e-disk', 'L-a-disk', 'L-x-disk'),
        func=lambda e, a, x, *_, **kw: e+a+x,
        plot_name_kwargs=dict(
            name=r"total disk neutrino luminosity",
            unit=r"$10^{51}\,\mathrm{erg\,s}^{-1}$",
            code_unit=r"$10^{51} M_\odot M_\odot^{-1}$"
        ),
        scale_factor=Units["Energy"]/Units["Time"]*1e3
    ),
    'mass-flow': dict(
        dependencies=("V^r", "dens"),
        func=_mass_flow,
        plot_name_kwargs=dict(
            name="mass flux",
            unit=r"$M_\odot$ ms$^{-1}$",
            code_unit="",
        ),
        reduction=sphere_surface_integral,
        scale_factor=1/Units['Time'],
    ),
    'mass-flow-cartoon': dict(
        dependencies=("V^r", "dens"),
        func=_mass_flow,
        plot_name_kwargs=dict(
            name="mass flux",
            unit=r"$M_\odot$ ms$^{-1}$",
            code_unit="",
        ),
        reduction=sphere_surface_integral,
        scale_factor=RUnits['Time'],
    ),
    'rho-bulk': dict(
        dependencies=("dens", "rho"),
        func=lambda *ar, **kw: 1,
        plot_name_kwargs=dict(
            name="density on bulk surface",
            unit="g cm$^{-3}$",
            code_unit=r"$M_\odot^{-2}$"
        ),
        kwargs=dict(
            func='log'
        ),
        reduction=_rho_bulk,
        scale_factor='Rho',
    ),
    'mass-bulk': dict(
        dependencies=("dens", "rho", "rho-bulk"),
        func=lambda dens, rho, rho_ns, *_, **kw: 2 *
        dens*(rho >= rho_ns).astype(int),
        plot_name_kwargs=dict(
            name="bulk mass",
            unit=r"$M_\odot$"
        ),
        reduction=integral,
    ),
    'mass-NS-bulk': dict(
        dependencies=("dens", "rho", "rho-bulk"),
        func=lambda dens, rho, rho_ns, *_, **kw: 2 *
        dens*(rho >= rho_ns/15).astype(int),
        plot_name_kwargs=dict(
            name="bulk mass",
            unit=r"$M_\odot$"
        ),
        reduction=integral,
    ),
    'volume-bulk': dict(
        dependencies=("reduce-weights", "rho", "rho-bulk"),
        func=lambda rw, rho, rho_ns, *_, **kw: 2 *
        rw*(rho >= rho_ns).astype(int),
        plot_name_kwargs=dict(
            name="bulk volume",
            unit="km$^3$",
            code_unit=r"$M_\odot^3$"
        ),
        reduction=integral,
        scale_factor=Units['Length']**3,
    ),
    'compactness-bulk': dict(
        dependencies=("rho-bulk", "mass-bulk", "volume-bulk"),
        func=lambda rb, m, v, *_, **kw: m/v**0.33333333333,
        plot_name_kwargs=dict(
            name="bulk compactness",
        ),
        save=False,
    ),
    'M-ejb-in-radius': dict(
        dependencies=("ejb-dens",),
        func=_ident,
        plot_name_kwargs=dict(
            name=r"ejected mass ($inner_ \leq r \leq outer_r)",
            unit=r"$M_\odot$",
            format_opt=dict(
                inner_r=_radius_format_func,
                outer_r=_radius_format_func
            ),
        ),
        reduction=integral,
        PPkeys=dict(inner_r=300, outer_r=1000),
    ),
    'M-ejg-in-radius': dict(
        dependencies=("ejg-dens",),
        func=_ident,
        plot_name_kwargs=dict(
            name=r"ejected mass ($inner_ \leq r \leq outer_r)",
            unit=r"$M_\odot$",
            format_opt=dict(
                inner_r=_radius_format_func,
                outer_r=_radius_format_func
            ),
        ),
        reduction=integral,
        PPkeys=dict(inner_r=300, outer_r=1000),
    ),
    'mass-in-rho-cont': dict(
        dependencies=("dens", "rho"),
        func=lambda dens, rho, *_, rho_cont=1e13 *
        RUnits['Rho'], **kw: 2*dens*(rho >= rho_cont),
        plot_name_kwargs=dict(
            name=r"mass ($\rho \geq rho_cont)",
            unit=r"$M_\odot$",
            format_opt=dict(
                rho_cont=_rho_cont_format_func
            ),
        ),
        reduction=integral,
        PPkeys=dict(rho_cont=1e13*RUnits["Rho"]),
    ),
    'J-in-rho-cont': dict(
        dependencies=("J_phi", "rho"),
        func=lambda jj, rho, *_, rho_cont=1e13 *
        RUnits['Rho'], **kw: 2*jj*(rho >= rho_cont),
        plot_name_kwargs=dict(
            name=r"J ($\rho > rho_cont)",
            format_opt=dict(
                rho_cont=_rho_cont_format_func
            ),
        ),
        reduction=integral,
        PPkeys=dict(rho_cont=1e13*RUnits["Rho"]),
    ),
    'J-out-rho-cont': dict(
        dependencies=("J_phi", "rho"),
        func=lambda jj, rho, *_, rho_cont=1e13 *
        RUnits['Rho'], **kw: 2*jj*(rho < rho_cont),
        plot_name_kwargs=dict(
            name=r"J ($\rho < rho_cont)",
            format_opt=dict(
                rho_cont=_rho_cont_format_func
            ),
        ),
        reduction=integral,
        PPkeys=dict(rho_cont=1e13*RUnits["Rho"]),
    ),
    'J/M-out-rho-cont': dict(
        dependencies=("J-out-rho-cont", "mass-out-rho-cont"),
        func=lambda jj, mm, *_, **kw: jj/mm,
        plot_name_kwargs=dict(
            name=r"J/M ($\rho < rho_cont)",
            format_opt=dict(
                rho_cont=_rho_cont_format_func
            ),
        ),
        PPkeys=dict(rho_cont=1e13*RUnits["Rho"]),
    ),
    'J/M-in-rho-cont': dict(
        dependencies=("J-in-rho-cont", "mass-in-rho-cont"),
        func=lambda jj, mm, *_, **kw: jj/mm,
        plot_name_kwargs=dict(
            name=r"J/M ($\rho > rho_cont)",
            format_opt=dict(
                rho_cont=_rho_cont_format_func
            ),
        ),
        PPkeys=dict(rho_cont=1e13*RUnits["Rho"]),
    ),
    'E-th-out-rho-cont': dict(
        dependencies=("e-th-eos", "rho"),
        func=lambda eth, rho, *_, rho_cont=1e13 *
        RUnits['Rho'], **kw: 2*eth*(rho < rho_cont).astype(int),
        plot_name_kwargs=dict(
            name=r"thermal energy ($\rho < rho_cont)",
            unit=r"$M_\odot$",
            format_opt=dict(
                rho_cont=_rho_cont_format_func
            ),
        ),
        reduction=integral,
        PPkeys=dict(rho_cont=1e13*RUnits["Rho"]),
    ),
    'E-th-in-rho-cont': dict(
        dependencies=("e-th-eos", "rho"),
        func=lambda eth, rho, *_, rho_cont=1e13 *
        RUnits['Rho'], **kw: 2*eth*(rho >= rho_cont).astype(int),
        plot_name_kwargs=dict(
            name=r"thermal energy ($\rho \geq rho_cont)",
            unit=r"$M_\odot$",
            format_opt=dict(
                rho_cont=_rho_cont_format_func
            ),
        ),
        reduction=integral,
        PPkeys=dict(rho_cont=1e13*RUnits["Rho"]),
    ),
    'E-th/M-in-rho-cont': dict(
        dependencies=("E-th-in-rho-cont", "mass-in-rho-cont"),
        func=lambda E, M, *_, **kw: E/M*100,
        save=False,
        plot_name_kwargs=dict(
            name=r"$\frac{E_{\rm th}}{M_{\rm HMNS}}$\n($\rho \geq rho_cont)",
            unit='%',
            format_opt=dict(
                rho_cont=_rho_cont_format_func
            ),
        ),
        PPkeys=dict(rho_cont=1e13*RUnits["Rho"]),
    ),
    'volume-in-rho-cont': dict(
        dependencies=("reduce-weights", "rho"),
        func=lambda rw, rho, *_, rho_cont=1e13 *
        RUnits['Rho'], **kw: 2*rw*(rho >= rho_cont).astype(int),
        plot_name_kwargs=dict(
            name=r"volume ($\rho \geq rho_cont)",
            unit="km $^3$",
            code_unit=r"$M_\odot^3$",
            format_opt=dict(
                rho_cont=_rho_cont_format_func
            ),
        ),
        reduction=integral,
        scale_factor=Units['Length']**3,
        PPkeys=dict(rho_cont=1e13*RUnits["Rho"]),
    ),
    'av-dens-in-rho-cont': dict(
        dependencies=("mass-in-rho-cont", "volume-in-rho-cont"),
        func=lambda mass, vol, *_, **kw: mass/vol,
        plot_name_kwargs=dict(
            name=r"average rest-mass density ($\rho \geq rho_cont)",
            format_opt=dict(
                rho_cont=_rho_cont_format_func
            ),
        ),
        save=False,
        PPkeys=dict(rho_cont=1e13*RUnits["Rho"]),
    ),
    'compactness-rho-cont': dict(
        dependencies=("mass-in-rho-cont", "volume-in-rho-cont"),
        func=lambda mass, vol, *_, **kw: 1.611992*mass/vol**(0.333333333),
        plot_name_kwargs=dict(
            name=r"compactness ($\rho \geq rho_cont)",
            format_opt=dict(
                rho_cont=_rho_cont_format_func
            ),
        ),
        save=False,
        PPkeys=dict(rho_cont=1e13*RUnits["Rho"]),
    ),
    'mass-out-rho-cont': dict(
        dependencies=("baryon-mass", "mass-in-rho-cont"),
        func=lambda Mtot, Min, *_, **kw: Mtot-Min,
        plot_name_kwargs=dict(
            name=r"mass ($\rho < rho_cont)",
            unit=r"$M_\odot$",
            format_opt=dict(
                rho_cont=_rho_cont_format_func
            ),
        ),
        save=False,
        PPkeys=dict(rho_cont=1e13*RUnits["Rho"]),
    ),
    'mass-disk-bulk': dict(
        dependencies=("baryon-mass", "mass-NS-bulk"),
        func=lambda mtot, mns, *_, **kw: mtot-mns,
        plot_name_kwargs=dict(
            name="disk mass",
            unit=r"$M_\odot$"
        ),
        save=False,
    ),
    'M-ej-esc-dot': dict(
        dependencies=("V^r", "dens", "u_t"),
        func=lambda vr, dens, u_t, *_, **kw: _mass_flow(vr, dens) * (u_t < -1),
        plot_name_kwargs=dict(
            name=r"$\dot{M}_{\rm ej}$ ($r=$radius)",
            unit=r"$M_\odot$ ms$^{-1}$",
            code_unit="",
            format_opt=dict(
                radius=lambda radius, code_units:
                (f"{radius:.0f} " + '$M_\\odot$' if code_units
                 else f"{radius*Units['Length']:.0f} km")
            ),
        ),
        reduction=sphere_surface_integral,
        scale_factor=RUnits['Time'],
        PPkeys=dict(radius=1000),
    ),
    'M-ej-esc': dict(
        dependencies=("M-ej-esc-dot",),
        func=_time_int,
        plot_name_kwargs=dict(
            name=r"$M_{\rm ej, esc}$ ($r=$radius)",
            unit=r"$M_\odot$",
            format_opt=dict(
                radius=lambda radius, code_units:
                (f"{radius:.0f} " + '$M_\\odot$' if code_units
                 else f"{radius*Units['Length']:.0f} km")
            ),
        ),
        save=False,
        PPkeys=dict(radius=1000),
    ),
    'M-ej-in': dict(
        dependencies=("dens", "u_t"),
        func=lambda dens, ut, *_, **kw: 2*dens*(ut <= -1),
        plot_name_kwargs=dict(
            name=r"$M_{\rm ej, in}$ ($r=$radius)",
            unit=r"$M_\odot$",
            format_opt=dict(
                radius=lambda radius, code_units:
                (f"{radius:.0f} " + '$M_\\odot$' if code_units
                 else f"{radius*Units['Length']:.0f} km")
            ),
        ),
        reduction=integral,
        PPkeys=dict(radius=1000),
    ),
    'M-ej-tot': dict(
        dependencies=("M-ej-esc", "M-ej-in"),
        func=lambda m_out, m_in, *_, **kw: m_out+m_in,
        plot_name_kwargs=dict(
            name=r"$M_{\rm ej, tot}$ ($r=$radius)",
            unit=r"$M_\odot$",
            format_opt=dict(
                radius=lambda radius, code_units:
                (f"{radius:.0f} " + '$M_\\odot$' if code_units
                 else f"{radius*Units['Length']:.0f} km")
            ),
        ),
        save=False,
        PPkeys=dict(radius=1000),
    ),
    'M-ejb-esc-dot': dict(
        dependencies=("V^r", "dens", "h-u_t"),
        func=lambda vr, dens, hu_t, *_, **__:
        _mass_flow(vr, dens) * (hu_t < -1),
        plot_name_kwargs=dict(
            name=r"$\dot{M}_{\rm ej}$ ($r=$radius)",
            unit=r"$M_\odot$ ms$^{-1}$",
            code_unit="",
            format_opt=dict(
                radius=lambda radius, code_units:
                (f"{radius:.0f} " + '$M_\\odot$' if code_units
                 else f"{radius*Units['Length']:.0f} km")
            ),
        ),
        reduction=sphere_surface_integral,
        scale_factor=RUnits['Time'],
        PPkeys=dict(radius=1000),
    ),
    'M-ejb-esc': dict(
        dependencies=("M-ejb-esc-dot",),
        func=_time_int,
        plot_name_kwargs=dict(
            name=r"$M_{\rm ej, esc}$ ($r=$radius)",
            unit=r"$M_\odot$",
            format_opt=dict(
                radius=lambda radius, code_units:
                (f"{radius:.0f} " + '$M_\\odot$' if code_units
                 else f"{radius*Units['Length']:.0f} km")
            ),
        ),
        save=False,
        PPkeys=dict(radius=1000),
    ),
    'Mp-ejb-esc-dot': dict(
        dependencies=("V^r", "dens", "ye", "h-u_t"),
        func=lambda vr, dens, ye, hu_t, *_, **__:
        _mass_flow(vr, dens*ye) * (hu_t < -1),
        plot_name_kwargs=dict(
            name=r"$\dot{M_p}_{\rm ej}$ ($r=$radius)",
            unit=r"$M_\odot$ ms$^{-1}$",
            code_unit="",
            format_opt=dict(
                radius=lambda radius, code_units:
                (f"{radius:.0f} " + '$M_\\odot$' if code_units
                 else f"{radius*Units['Length']:.0f} km")
            ),
        ),
        reduction=sphere_surface_integral,
        scale_factor=RUnits['Time'],
        PPkeys=dict(radius=1000),
    ),
    'Mp-ejb-esc': dict(
        dependencies=("Mp-ejb-esc-dot",),
        func=_time_int,
        plot_name_kwargs=dict(
            name=r"$M_{p, {\rm ej, esc}}$ ($r=$radius)",
            unit=r"$M_\odot$",
            format_opt=dict(
                radius=lambda radius, code_units:
                (f"{radius:.0f} " + '$M_\\odot$' if code_units
                 else f"{radius*Units['Length']:.0f} km")
            ),
        ),
        save=False,
        PPkeys=dict(radius=1000),
    ),
    'ye-ejb-esc-tot': dict(
        dependencies=("Mp-ejb-esc", "M-ejb-esc",),
        func=lambda Mp, M, *_, **kw: Mp/M,
        plot_name_kwargs=dict(
            name=r"$Y_{e, {\rm ej, esc}}$ (cumulative) ($r=$radius)",
            format_opt=dict(
                radius=lambda radius, code_units:
                (f"{radius:.0f} " + '$M_\\odot$' if code_units
                 else f"{radius*Units['Length']:.0f} km")
            ),
        ),
        save=False,
        PPkeys=dict(radius=1000),
    ),
    'ye-ejb-esc': dict(
        dependencies=("Mp-ejb-esc-dot", "M-ejb-esc-dot",),
        func=lambda Mp, M, *_, **kw: Mp/M,
        plot_name_kwargs=dict(
            name=r"$Y_{e, {\rm ej, esc}}$ ($r=$radius)",
            format_opt=dict(
                radius=lambda radius, code_units:
                (f"{radius:.0f} " + '$M_\\odot$' if code_units
                 else f"{radius*Units['Length']:.0f} km")
            ),
        ),
        save=False,
        PPkeys=dict(radius=1000),
    ),
    'M-ejb-in': dict(
        dependencies=("dens", "u_t", 'h'),
        func=lambda dens, ut, h, *_, **kw: 2*dens*(h*ut <= -1),
        plot_name_kwargs=dict(
            name=r"$M_{\rm ej, in}$ ($r=$radius)",
            unit=r"$M_\odot$",
            format_opt=dict(
                radius=lambda radius, code_units:
                (f"{radius:.0f} " + '$M_\\odot$' if code_units
                 else f"{radius*Units['Length']:.0f} km")
            ),
        ),
        reduction=integral,
        PPkeys=dict(radius=1000),
    ),
    'M-ejb-tot': dict(
        dependencies=("M-ejb-esc", "M-ejb-in"),
        func=lambda m_out, m_in, *_, **kw: m_out+m_in,
        plot_name_kwargs=dict(
            name=r"$M_{\rm ej, tot}$ ($r=$radius)",
            unit=r"$M_\odot$",
            format_opt=dict(
                radius=lambda radius, code_units:
                (f"{radius:.0f} " + '$M_\\odot$' if code_units
                 else f"{radius*Units['Length']:.0f} km")
            ),
        ),
        save=False,
        PPkeys=dict(radius=1000),
    ),
    'temp-bulk-mean': dict(
        dependencies=("temp", "rho", "rho-bulk"),
        func=lambda temp, rho, rbulk, *_, **kw: temp*(rho >= rbulk),
        plot_name_kwargs=dict(
            name=r"mean bulk temperature",
            unit="MeV",
        ),
        reduction=mean,
    ),
    'ye-bulk-mean': dict(
        dependencies=("ye", "rho", "rho-bulk"),
        func=lambda ye, rho, rbulk, *_, **kw: ye*(rho >= rbulk),
        plot_name_kwargs=dict(
            name=r"mean bulk $Y_e$",
        ),
        reduction=mean,
    ),
    'entr-bulk-mean': dict(
        dependencies=("entr", "rho", "rho-bulk"),
        func=lambda entr, dens, dns, *_, **kw: entr*(dens >= dns),
        plot_name_kwargs=dict(
            name=r"mean bulk entropy",
            unit=r"$k_{\rm B}$/nuc.",
        ),
        reduction=mean,
    ),
    'press-bulk-mean': dict(
        dependencies=("press", "rho", "rho-bulk"),
        func=lambda press, dens, dns, *_, **kw: press*(dens >= dns),
        plot_name_kwargs=dict(
            name=r"mean bulk pressure",
            code_unit="$M_\\odot^{-2}$",
            unit="g cm$^{-1}$ s$^{-2}$",
        ),
        reduction=mean,
        scale_factor="Press",
    ),
    'temp-disk-mean': dict(
        dependencies=("temp", "rho", "rho-bulk"),
        func=lambda temp, dens, dns, *_, **kw: temp*(dens < dns),
        plot_name_kwargs=dict(
            name=r"mean disk temperature",
            unit="MeV",
        ),
        reduction=mean,
    ),
    'ye-disk-mean': dict(
        dependencies=("ye", "rho", "rho-bulk"),
        func=lambda ye, dens, dns, *_, **kw: ye*(dens < dns),
        plot_name_kwargs=dict(
            name=r"mean disk $Y_e$",
        ),
        reduction=mean,
    ),
    'entr-disk-mean': dict(
        dependencies=("entr", "rho", "rho-bulk"),
        func=lambda entr, dens, dns, *_, **kw: entr*(dens < dns),
        plot_name_kwargs=dict(
            name=r"mean disk entropy",
            unit=r"$k_{\rm B}$/nuc.",
        ),
        reduction=mean,
    ),
    'press-disk-mean': dict(
        dependencies=("press", "rho", "rho-bulk"),
        func=lambda press, dens, dns, *_, **kw: press*(dens < dns),
        plot_name_kwargs=dict(
            name=r"mean disk pressure",
            code_unit="$M_\\odot^{-2}$",
            unit="g cm$^{-1}$ s$^{-2}$",
        ),
        reduction=mean,
        scale_factor="Press",
    ),
    'temp-in-rho-cont-mean': dict(
        dependencies=("temp", "rho",),
        func=lambda temp, rho, rho_cont=1e13 *
        RUnits['Rho'], *_, **kw: temp*_nan_mask(rho >= rho_cont),
        plot_name_kwargs=dict(
            name=r"mean temperature ($\rho \geq rho_cont)",
            unit="MeV",
            format_opt=dict(
                rho_cont=_rho_cont_format_func
            ),
        ),
        reduction=mean,
        PPkeys=dict(rho_cont=1e13*RUnits["Rho"]),
    ),
    'ye-in-rho-cont-mean': dict(
        dependencies=("ye", "rho",),
        func=lambda ye, rho, rho_cont=1e13 *
        RUnits['Rho'], *_, **kw: ye*_nan_mask(rho >= rho_cont),
        plot_name_kwargs=dict(
            name=r"mean $Y_e$ ($\rho \geq rho_cont)",
            format_opt=dict(
                rho_cont=_rho_cont_format_func
            ),
        ),
        reduction=mean,
        PPkeys=dict(rho_cont=1e13*RUnits["Rho"]),
    ),
    'entr-in-rho-cont-mean': dict(
        dependencies=("entr", "rho",),
        func=lambda entr, rho, rho_cont=1e13 *
        RUnits["Rho"], *_, **kw: entr*_nan_mask(rho >= rho_cont),
        plot_name_kwargs=dict(
            name=r"mean entropy ($\rho \geq rho_cont)",
            unit=r"$k_{\rm B}$/nuc.",
            format_opt=dict(
                rho_cont=_rho_cont_format_func
            ),
        ),
        reduction=mean,
        PPkeys=dict(rho_cont=1e13*RUnits["Rho"]),
    ),
    'press-in-rho-cont-mean': dict(
        dependencies=("press", "rho",),
        func=lambda press, rho, rho_cont=1e13 *
        RUnits["Rho"], *_, **kw: press*_nan_mask(rho >= rho_cont),
        plot_name_kwargs=dict(
            name=r"mean pressure ($\rho \geq rho_cont)",
            code_unit="$M_\\odot^{-2}$",
            unit="g cm$^{-1}$ s$^{-2}$",
            format_opt=dict(
                rho_cont=_rho_cont_format_func
            ),
        ),
        reduction=mean,
        PPkeys=dict(rho_cont=1e13*RUnits["Rho"]),
        scale_factor="Press",
    ),
    'temp-out-rho-cont-mean': dict(
        dependencies=("temp", "rho",),
        func=_mean,
        plot_name_kwargs=dict(
            name=r"mean temperature (rho_contrho2_cont)",
            unit="MeV",
            format_opt=dict(
                rho_cont=_rho_cont_format_func,
                rho2_cont=lambda rho2_cont=None, code_units=False:
                "" if rho2_cont is None else
                (f", $\\rho > {rho2_cont:.0f} " + r'M_\odot^{-2}$' if code_units else
                 f", $\\rho > {rho2_cont*Units['Rho']:.0e}"+r"\,$g cm$^{-3}$")
            ),
        ),
        reduction=mean,
        PPkeys=dict(rho_cont=1e13*RUnits["Rho"], rho2_cont=None),
    ),
    'ye-out-rho-cont-mean': dict(
        dependencies=("ye", "rho",),
        func=lambda ye, rho, rho_cont=1e13 *
        RUnits['Rho'], *_, **kw: ye*_nan_mask(rho < rho_cont),
        plot_name_kwargs=dict(
            name=r"mean $Y_e$ ($\rho < rho_cont)",
            format_opt=dict(
                rho_cont=_rho_cont_format_func
            ),
        ),
        reduction=mean,
        PPkeys=dict(rho_cont=1e13*RUnits["Rho"]),
    ),
    'entr-out-rho-cont-mean': dict(
        dependencies=("entr", "rho",),
        func=lambda entr, rho, rho_cont=1e13 *
        RUnits["Rho"], *_, **kw: entr*_nan_mask(rho < rho_cont),
        plot_name_kwargs=dict(
            name=r"mean entropy ($\rho < rho_cont)",
            unit=r"$k_{\rm B}$/nuc.",
            format_opt=dict(
                rho_cont=_rho_cont_format_func
            ),
        ),
        reduction=mean,
        PPkeys=dict(rho_cont=1e13*RUnits["Rho"]),
    ),
    'press-out-rho-cont-mean': dict(
        dependencies=("press", "rho",),
        func=lambda press, rho, rho_cont=1e13 *
        RUnits["Rho"], *_, **kw: press*_nan_mask(rho < rho_cont),
        plot_name_kwargs=dict(
            name=r"mean pressure ($\rho < rho_cont)",
            code_unit="$M_\\odot^{-2}$",
            unit="g cm$^{-1}$ s$^{-2}$",
            format_opt=dict(
                rho_cont=_rho_cont_format_func
            ),
        ),
        reduction=mean,
        PPkeys=dict(rho_cont=1e13*RUnits["Rho"]),
        scale_factor="Press",
    ),
    'entr-min': dict(
        dependencies=("entr", ),
        func=lambda entr, *_, **kw: entr,
        plot_name_kwargs=dict(
            name=r"minumum entropy",
            unit=r"$k_{\rm B}$/nuc.",
        ),
        reduction=minimum,
    ),
    'eps-max': dict(
        dependencies=("eps",),
        func=lambda press, *_, **kw: press,
        plot_name_kwargs=dict(
            name=r"maximum internal energy",
        ),
        reduction=maximum,
    ),
    'press-max': dict(
        dependencies=("press",),
        func=lambda press, *_, **kw: press,
        plot_name_kwargs=dict(
            name=r"maximum pressure",
            code_unit="$M_\\odot^{-2}$",
            unit="g cm$^{-1}$ s$^{-2}$",
        ),
        reduction=maximum,
        scale_factor="Press",
    ),
    'Gamma-th-max': dict(
        dependencies=("Gamma-th",),
        func=lambda Gamma, *_, **kw: Gamma,
        plot_name_kwargs=dict(
            name=r"maximum $\Gamma_{\rm th}$",
        ),
        reduction=maximum,
    ),
    'Omega-max': dict(
        dependencies=("Omega-excised",),
        func=lambda Gamma, *_, **kw: Gamma,
        reduction=maximum,
        plot_name_kwargs=dict(
            name="$\Omega$",
            unit="rad ms$^{-1}$",
            code_unit="rad $M_\\odot^{-1}$",
        ),
        scale_factor=RUnits['Time']
    ),
    'Omega-mean': dict(
        dependencies=("Omega-excised",),
        func=lambda Gamma, *_, **kw: Gamma,
        reduction=mean,
        plot_name_kwargs=dict(
            name="$<\Omega>$",
            unit="rad ms$^{-1}$",
            code_unit="rad $M_\\odot^{-1}$",
        ),
        scale_factor=RUnits['Time']
    ),
    'temp-max-pp': dict(
        dependencies=("temp",),
        func=lambda temp, *_, **kw: temp,
        reduction=maximum,
        plot_name_kwargs=dict(
            name="$T$",
            unit="MeV",
        ),
    ),
    '_press-weight': dict(
        dependencies=("press-th/cold-eos", "rho"),
        func=lambda pratio, rho, rho_cont=1e13 *
        RUnits['Rho'], *_, **kw: _nan_mask(rho >= rho_cont)*pratio,
        plot_name_kwargs=dict(
            name=r"aux. thermal pressure weight",
            format_opt=dict(
                rho_cont=_rho_cont_format_func),
        ),
        reduction=mean,
        PPkeys=dict(rho_cont=1e13*RUnits["Rho"]),
    ),
    'Gamma-th-out-rho-cont-mean': dict(
        dependencies=("Gamma-th", "rho", ),
        func=lambda Gamma, rho, rho_cont=1e13 *
        RUnits['Rho'], *_, **kw: Gamma*_nan_mask(rho < rho_cont),
        plot_name_kwargs=dict(
            name=r"mean $\Gamma_{\rm th}$ ($\rho < rho_cont)",
            format_opt=dict(
                rho_cont=_rho_cont_format_func
            ),
        ),
        reduction=mean,
        PPkeys=dict(rho_cont=1e13*RUnits["Rho"]),
    ),
    'Gamma-th-in-rho-cont-mean': dict(
        dependencies=("Gamma-th", "rho"),
        func=lambda Gamma, rho, rho_cont=1e13 *
        RUnits['Rho'], *_, **kw: Gamma*_nan_mask(rho >= rho_cont),
        plot_name_kwargs=dict(
            name=r"mean $\Gamma_{\rm th}$ ($\rho > rho_cont)",
            format_opt=dict(
                rho_cont=_rho_cont_format_func
            ),
        ),
        reduction=mean,
        PPkeys=dict(rho_cont=1e13*RUnits["Rho"]),
    ),
    'Gamma-th-in-rho-cont-weighted': dict(
        dependencies=("Gamma-th", "rho", "press-th/cold-eos", "_press-weight"),
        func=lambda Gamma, rho, pratio, pweight, rho_cont=1e13 *
        RUnits['Rho'], *_, **kw: Gamma*_nan_mask(rho >= rho_cont)*pratio/pweight,
        plot_name_kwargs=dict(
            name=r"weighted mean $\Gamma_{\rm th}$",
            format_opt=dict(
                rho_cont=_rho_cont_format_func
            ),
        ),
        reduction=mean,
        PPkeys=dict(rho_cont=1e13*RUnits["Rho"]),
    ),
    'Gamma-th-eos-max': dict(
        dependencies=("Gamma-th-eos",),
        func=lambda Gamma, *_, **kw: Gamma,
        plot_name_kwargs=dict(
            name=r"maximum $\Gamma_{\rm th}$",
        ),
        reduction=maximum,
    ),
    'h-abs': dict(
        dependencies=('h+', 'hx'),
        func=lambda hp, hx, *_, **kw: np.abs(hp-1j*hx),
        save=False,
        plot_name_kwargs=dict(
            name="$|h^{ll mm}|$",
            format_opt=dict(
                ll=lambda ll, **_: str(ll),
                mm=lambda mm, **_: str(mm)
            )
        ),
        PPkeys=dict(ll=2, mm=2, power=1, n_points=3000, u_junk=200.)
    ),
    'm-decomp-r': dict(
        dependencies=("dens", ),
        func=_m_decomp_r,
        plot_name_kwargs=dict(
            name=r"Re $C_mm$",
            format_opt=dict(
                mm=lambda mm, *_, **__: str(mm)),
            unit=r"$M_\odot$"
        ),
        reduction=integral_2D,
        PPkeys=dict(mm=2),
    ),
    'm-decomp-i': dict(
        dependencies=("dens", ),
        func=_m_decomp_i,
        plot_name_kwargs=dict(
            name=r"Im $C_mm$",
            format_opt=dict(
                mm=lambda mm, *_, **__: str(mm)),
            unit=r"$M_\odot$"
        ),
        reduction=integral_2D,
        PPkeys=dict(mm=2),
    ),
    'm-decomp-abs': dict(
        dependencies=("m-decomp-r", "m-decomp-i"),
        func=lambda rr, ii, *_, **__: np.abs(rr + 1j*ii),
        plot_name_kwargs=dict(
            name=r"$|C_mm|$",
            format_opt=dict(
                mm=lambda mm, *_, **__: str(mm)),
            unit=r"$M_\odot$",
        ),
        kwargs=dict(
            func='log'
        ),
        PPkeys=dict(mm=2),
        save=False,
    ),
    'ye-nu-eq': dict(
        dependencies=("L-e-ene-M0", "L-a-ene-M0",
                      "L-e-num-M0", "L-a-num-M0", ),
        func=_ye_eq,
        plot_name_kwargs=dict(
            name=r"$Y_{e, {\rm eq}}$"
        ),
        save=False,
    ),
}
