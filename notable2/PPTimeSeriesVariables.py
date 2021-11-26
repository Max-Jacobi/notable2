import numpy as np
from scipy.integrate import cumtrapz  # type: ignore
from notable2.Reductions import integral, sphere_surface_integral
from notable2.Reductions import mean, minimum, maximum
from notable2.Utils import Units, RUnits


def _mass_flow(vx, vy, vz,
               bx, by, bz,
               gxx, gxy, gxz,
               gyy, gyz, gzz,
               alp, dens,
               x, y, z, **_):
    Vx = alp*vx - bx
    Vy = alp*vy - by
    Vz = alp*vz - bz
    r = (gxx*x**2 + gyy*y**2 + gzz*z**2 +
         2*gxy*x*y + 2*gxz*x*z + 2*gyz*y*z)**.5
    return dens*(gxx*Vx*x + gyy*Vy*y + gzz*Vz*z +
                 gxy*(Vx*y + Vy*x) +
                 gxz*(Vx*z + Vz*x) +
                 gyz*(Vy*z + Vz*y)
                 )/r


def _mass_flow_ej(vx, vy, vz,
                  bx, by, bz,
                  gxx, gxy, gxz,
                  gyy, gyz, gzz,
                  alp, dens, u_t,
                  x, y, z, **_):
    return _mass_flow(vx, vy, vz, bx, by, bz,
                      gxx, gxy, gxz, gyy, gyz, gzz,
                      alp, dens, x, y, z) * (u_t < -1)


def _mass_flow_cartoon(vx, vz, bx, bz,
                       gxx, gxz, gzz,
                       alp, dens,
                       x, z, **_):
    Vx = alp*vx - bx
    Vz = alp*vz - bz
    r = (gxx*x**2 + gzz*z**2 + 2*gxz*x*z)**.5
    return dens*(gxx*Vx*x + gzz*Vz*z + gxz*(Vx*z + Vz*x))/r


def _times_domain_volume(dependencies, its, var, func, **kwargs):
    bmass = dependencies[0].get_data(it=its)
    coords = var.sim.get_coords('xyz', 0, exclude_ghosts=3)[0]
    vol = np.prod([cc[-1] - cc[0] for cc in coords.values()])
    return func(bmass.data)*vol


def _rho_bulk(dependencies, its, var, **kwargs):
    region = 'xz' if var.sim.is_cartoon else 'xyz'
    rl = var.sim.rls.max()
    result = np.zeros_like(its, dtype=float)

    n_its = len(its)
    for ii, it in enumerate(its):
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
    'L-e': dict(
        backups=["L-e-pp"],
        dependencies=('L-e-dummy',),
        func=lambda dd, *_, **kw: 2*dd,
        plot_name_kwargs=dict(
            name=r"electron neutrino luminosity",
            unit=r"$10^{51}\,\mathrm{erg\,s}^{-1}$",
            code_unit=r"$10^{51} m_\odot m_\odot^{-1}$"
        ),
        reduction=_times_domain_volume,
        scale_factor=Units["Energy"]/Units["Time"]*1e3
    ),
    'L-a': dict(
        backups=["L-a-pp"],
        dependencies=('L-a-dummy',),
        func=lambda dd, *_, **kw: 2*dd,
        plot_name_kwargs=dict(
            name=r"electron antineutrino luminosity",
            unit=r"$10^{51}\,\mathrm{erg\,s}^{-1}$",
            code_unit=r"$10^{51} M_\odot M_\odot^{-1}$"
        ),
        reduction=_times_domain_volume,
        scale_factor=Units["Energy"]/Units["Time"]*1e3
    ),
    'L-x': dict(
        backups=["L-pp"],
        dependencies=('L-x-dummy',),
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
        dependencies=("vel^x", "vel^y", "vel^z",
                      "beta^x", "beta^y", "beta^z",
                      'g_xx', 'g_xy', 'g_xz',
                      'g_yy', 'g_yz', 'g_zz',
                      "alpha", "dens"),
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
        dependencies=("vel^x", "vel^z", "beta^x", "beta^z",
                      'g_xx', 'g_xz', 'g_zz',
                      "alpha", "dens"),
        func=_mass_flow_cartoon,
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
        func=lambda dens, rho, rho_ns, *_, **kw: 2*dens*(rho >= rho_ns).astype(int),
        plot_name_kwargs=dict(
            name="bulk mass",
            unit=r"$M_\odot$"
        ),
        reduction=integral,
    ),
    'volume-bulk': dict(
        dependencies=("reduce-weights", "rho", "rho-bulk"),
        func=lambda rw, rho, rho_ns, *_, **kw: 2*rw*(rho >= rho_ns).astype(int),
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
    'mass-in-rho-cont': dict(
        dependencies=("dens", "rho"),
        func=lambda dens, rho, *_, rho_cont=1e13*RUnits['Rho'], **kw: 2*dens*(rho >= rho_cont).astype(int),
        plot_name_kwargs=dict(
            name=r"mass ($\rho \geq rho_cont)",
            unit=r"$M_\odot$",
            format_func=dict(
                rho_cont=lambda rho_cont=1e13*RUnits['Rho'], code_units=False:
                (f"{rho_cont:.0f} " + r'M_\odot^{-2}$' if code_units
                 else f"{rho_cont*Units['Rho']:.0e}"+r"\,$g cm$^{-3}$")
            ),
        ),
        reduction=integral,
        PPkeys=['rho_cont'],
    ),
    'volume-in-rho-cont': dict(
        dependencies=("reduce-weights", "rho"),
        func=lambda rw, rho, *_, rho_cont=1e13*RUnits['Rho'], **kw: 2*rw*(rho >= rho_cont).astype(int),
        plot_name_kwargs=dict(
            name=r"volume ($\rho \geq rho_cont)",
            unit="km $^3$",
            code_unit=r"$M_\odot^3$",
            format_func=dict(
                rho_cont=lambda rho_cont=1e13*RUnits['Rho'], code_units=False:
                (f"{rho_cont:.0f} " + r'M_\odot^{-2}$' if code_units
                 else f"{rho_cont*Units['Rho']:.0e}"+r"\,$g cm$^{-3}$")
            ),
        ),
        reduction=integral,
        scale_factor=Units['Length']**3,
        PPkeys=['rho_cont'],
    ),
    'compactness-rho-cont': dict(
        dependencies=("mass-in-rho-cont", "volume-in-rho-cont"),
        func=lambda mass, vol, *_, **kw: mass/vol**(0.333333333),
        plot_name_kwargs=dict(
            name=r"compactness ($\rho \geq rho_cont)",
            format_func=dict(
                rho_cont=lambda rho_cont=1e13*RUnits['Rho'], code_units=False:
                (f"{rho_cont:.1e} " + r'M_\odot^{-2}$' if code_units
                 else f"{rho_cont*Units['Rho']:.1e}"+r"\,$g cm$^{-3}$"),
            ),
        ),
        PPkeys=['rho_cont'],
    ),
    'mass-out-rho-cont': dict(
        dependencies=("baryon-mass", "mass-in-rho-cont"),
        func=lambda Mtot, Min, *_, **kw: Mtot-Min,
        plot_name_kwargs=dict(
            name=r"mass ($\rho < rho_cont)",
            unit=r"$M_\odot$",
            format_func=dict(
                rho_cont=lambda rho_cont=1e13*RUnits['Rho'], code_units=False:
                (f"{rho_cont:.0f} " + r'M_\odot^{-2}$' if code_units
                 else f"{rho_cont*Units['Rho']:.0e}"+r"\,$g cm$^{-3}$")
            ),
        ),
        save=False,
        PPkeys=['rho_cont'],
    ),
    'mass-disk': dict(
        dependencies=("baryon-mass", "mass-bulk"),
        func=lambda mtot, mns, *_, **kw: mtot-mns,
        plot_name_kwargs=dict(
            name="disk mass",
            unit=r"$M_\odot$"
        ),
        save=False,
    ),
    'M-ej-esc-dot': dict(
        dependencies=("vel^x", "vel^y", "vel^z",
                      "beta^x", "beta^y", "beta^z",
                      'g_xx', 'g_xy', 'g_xz',
                      'g_yy', 'g_yz', 'g_zz',
                      "alpha", "dens", "u_t"),
        func=_mass_flow_ej,
        plot_name_kwargs=dict(
            name=r"$\dot{M}_{\rm ej}$ ($r=$radius)",
            unit=r"$M_\odot$ ms$^{-1}$",
            code_unit="",
            format_func=dict(
                radius=lambda radius, code_units:
                (f"{radius:.0f} " + '$M_\\odot$' if code_units
                 else f"{radius*Units['Length']:.0f} km")
            ),
        ),
        reduction=sphere_surface_integral,
        scale_factor=RUnits['Time'],
        PPkeys=['radius'],
    ),
    'M-ej-esc': dict(
        dependencies=("M-ej-esc-dot",),
        func=_time_int,
        plot_name_kwargs=dict(
            name=r"$M_{\rm ej, esc}$ ($r=$radius)",
            unit=r"$M_\odot$",
            format_func=dict(
                radius=lambda radius, code_units:
                (f"{radius:.0f} " + '$M_\\odot$' if code_units
                 else f"{radius*Units['Length']:.0f} km")
            ),
        ),
        save=False,
        PPkeys=['radius'],
    ),
    'M-ej-in': dict(
        dependencies=("dens", "u_t"),
        func=lambda dens, ut, *_, **kw: 2*dens*(ut <= -1),
        plot_name_kwargs=dict(
            name=r"$M_{\rm ej, in}$ ($r=$radius)",
            unit=r"$M_\odot$",
            format_func=dict(
                radius=lambda radius, code_units:
                (f"{radius:.0f} " + '$M_\\odot$' if code_units
                 else f"{radius*Units['Length']:.0f} km")
            ),
        ),
        reduction=integral,
        PPkeys=['radius'],
    ),
    'M-ej-tot': dict(
        dependencies=("M-ej-esc", "M-ej-in"),
        func=lambda m_out, m_in, *_, **kw: m_out+m_in,
        plot_name_kwargs=dict(
            name=r"$M_{\rm ej, tot}$ ($r=$radius)",
            unit=r"$M_\odot$",
            format_func=dict(
                radius=lambda radius, code_units:
                (f"{radius:.0f} " + '$M_\\odot$' if code_units
                 else f"{radius*Units['Length']:.0f} km")
            ),
        ),
        save=False,
        PPkeys=['radius'],
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
        func=lambda temp, rho, rho_cont=1e13*RUnits['Rho'], *_, **kw: temp*(rho >= rho_cont),
        plot_name_kwargs=dict(
            name=r"mean temperature ($\rho \geq rho_cont)",
            unit="MeV",
            format_func=dict(
                rho_cont=lambda rho_cont=1e13*RUnits['Rho'], code_units=False:
                (f"{rho_cont:.0f} " + r'M_\odot^{-2}$' if code_units
                 else f"{rho_cont*Units['Rho']:.0e}"+r"\,$g cm$^{-3}$")
            ),
        ),
        reduction=mean,
        PPkeys=['rho_cont'],
    ),
    'ye-in-rho-cont-mean': dict(
        dependencies=("ye", "rho",),
        func=lambda ye, rho, rho_cont=1e13*RUnits['Rho'], *_, **kw: ye*(rho >= rho_cont),
        plot_name_kwargs=dict(
            name=r"mean $Y_e$ ($\rho \geq rho_cont)",
            format_func=dict(
                rho_cont=lambda rho_cont=1e13*RUnits['Rho'], code_units=False:
                (f"{rho_cont:.0f} " + r'M_\odot^{-2}$' if code_units
                 else f"{rho_cont*Units['Rho']:.0e}"+r"\,$g cm$^{-3}$")
            ),
        ),
        reduction=mean,
        PPkeys=['rho_cont'],
    ),
    'entr-in-rho-cont-mean': dict(
        dependencies=("entr", "rho",),
        func=lambda entr, rho, rho_cont=1e13*RUnits["Rho"], *_, **kw: entr*(rho >= rho_cont),
        plot_name_kwargs=dict(
            name=r"mean entropy ($\rho \geq rho_cont)",
            unit=r"$k_{\rm B}$/nuc.",
            format_func=dict(
                rho_cont=lambda rho_cont=1e13*RUnits['Rho'], code_units=False:
                (f"{rho_cont:.0f} " + r'M_\odot^{-2}$' if code_units
                 else f"{rho_cont*Units['Rho']:.0e}"+r"\,$g cm$^{-3}$")
            ),
        ),
        reduction=mean,
        PPkeys=['rho_cont'],
    ),
    'press-in-rho-cont-mean': dict(
        dependencies=("press", "rho",),
        func=lambda press, rho, rho_cont=1e13*RUnits["Rho"], *_, **kw: press*(rho >= rho_cont),
        plot_name_kwargs=dict(
            name=r"mean pressure ($\rho \geq rho_cont)",
            code_unit="$M_\\odot^{-2}$",
            unit="g cm$^{-1}$ s$^{-2}$",
            format_func=dict(
                rho_cont=lambda rho_cont=1e13*RUnits['Rho'], code_units=False:
                (f"{rho_cont:.0f} " + r'M_\odot^{-2}$' if code_units
                 else f"{rho_cont*Units['Rho']:.0e}"+r"\,$g cm$^{-3}$")
            ),
        ),
        reduction=mean,
        PPkeys=['rho_cont'],
        scale_factor="Press",
    ),
    'temp-out-rho-cont-mean': dict(
        dependencies=("temp", "rho",),
        func=lambda temp, rho, rho_cont=1e13*RUnits['Rho'], *_, **kw: temp*(rho < rho_cont),
        plot_name_kwargs=dict(
            name=r"mean temperature ($\rho < rho_cont)",
            unit="MeV",
            format_func=dict(
                rho_cont=lambda rho_cont=1e13*RUnits['Rho'], code_units=False:
                (f"{rho_cont:.0f} " + r'M_\odot^{-2}$' if code_units
                 else f"{rho_cont*Units['Rho']:.0e}"+r"\,$g cm$^{-3}$")
            ),
        ),
        reduction=mean,
        PPkeys=['rho_cont'],
    ),
    'ye-out-rho-cont-mean': dict(
        dependencies=("ye", "rho",),
        func=lambda ye, rho, rho_cont=1e13*RUnits['Rho'], *_, **kw: ye*(rho < rho_cont),
        plot_name_kwargs=dict(
            name=r"mean $Y_e$ ($\rho < rho_cont)",
            format_func=dict(
                rho_cont=lambda rho_cont=1e13*RUnits['Rho'], code_units=False:
                (f"{rho_cont:.0f} " + r'M_\odot^{-2}$' if code_units
                 else f"{rho_cont*Units['Rho']:.0e}"+r"\,$g cm$^{-3}$")
            ),
        ),
        reduction=mean,
        PPkeys=['rho_cont'],
    ),
    'entr-out-rho-cont-mean': dict(
        dependencies=("entr", "rho",),
        func=lambda entr, rho, rho_cont=1e13*RUnits["Rho"], *_, **kw: entr*(rho < rho_cont),
        plot_name_kwargs=dict(
            name=r"mean entropy ($\rho < rho_cont)",
            unit=r"$k_{\rm B}$/nuc.",
            format_func=dict(
                rho_cont=lambda rho_cont=1e13*RUnits['Rho'], code_units=False:
                (f"{rho_cont:.0f} " + r'M_\odot^{-2}$' if code_units
                 else f"{rho_cont*Units['Rho']:.0e}"+r"\,$g cm$^{-3}$")
            ),
        ),
        reduction=mean,
        PPkeys=['rho_cont'],
    ),
    'press-out-rho-cont-mean': dict(
        dependencies=("press", "rho",),
        func=lambda press, rho, rho_cont=1e13*RUnits["Rho"], *_, **kw: press*(rho < rho_cont),
        plot_name_kwargs=dict(
            name=r"mean pressure ($\rho < rho_cont)",
            code_unit="$M_\\odot^{-2}$",
            unit="g cm$^{-1}$ s$^{-2}$",
            format_func=dict(
                rho_cont=lambda rho_cont=1e13*RUnits['Rho'], code_units=False:
                (f"{rho_cont:.0f} " + r'M_\odot^{-2}$' if code_units
                 else f"{rho_cont*Units['Rho']:.0e}"+r"\,$g cm$^{-3}$")
            ),
        ),
        reduction=mean,
        PPkeys=['rho_cont'],
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
}
