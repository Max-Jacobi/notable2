import numpy as np
from scipy.integrate import cumtrapz
from notable2.Reductions import integral, sphere_surface_integral
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


def _dens_ns(dependencies, its, var, **kwargs):
    region = 'xz' if var.sim.is_cartoon else 'xyz'
    rl = var.sim.rls.max()
    result = np.zeros_like(its, dtype=float)

    for ii, it in enumerate(its):
        if var.sim.verbose:
            print(f"{var.sim.sim_name} - {var.key}: Processing iteration {it} ({ii/len(its)*100:.1f}%)",
                  end=('\r' if var.sim.verbose == 1 else '\n'))
        dens = dependencies[0].get_data(region=region, it=it)
        dds = dens[rl]
        coords = dens.coords[rl]

        vols = np.ones_like(dds)
        if var.sim.is_cartoon:
            vols *= np.abs(coords['x'])

        dds = dds.ravel()
        vols = vols.ravel()
        mass = vols*dds

        fin_msk = np.isfinite(dds)
        mass = mass[fin_msk]
        vols = vols[fin_msk]
        dds = dds[fin_msk]

        isort = np.argsort(dds)[::-1]

        dds = dds[isort]
        mass = np.cumsum(mass[isort])
        vols = np.cumsum(vols[isort])
        Cs = mass/vols**.333333333

        result[ii] = dds[Cs.argmax()]
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
            unit="$M_\odot$"
        ),
        reduction=_times_domain_volume,
    ),
    'baryon-mass-pp': dict(
        dependencies=('dens',),
        func=lambda dd, *_, **kw: dd*2,
        plot_name_kwargs=dict(
            name="total baryon mass",
            unit="$M_\odot$"
        ),
        reduction=integral
    ),
    'L-e': dict(
        dependencies=('L-nu-e',),
        func=lambda dd, *_, **kw: dd,
        plot_name_kwargs=dict(
            name=r"electron neutrino luminosity",
            unit=r"$10^{51}\,\mathrm{erg\,s}^{-1}$",
            code_unit=r"$10^{51} M_\odot M_\odot^{-1}$"
        ),
        reduction=integral,
        scale_factor=Units["Energy"]/Units["Time"]*1e3
    ),
    'L-a': dict(
        dependencies=('L-nu-a',),
        func=lambda dd, *_, **kw: dd,
        plot_name_kwargs=dict(
            name=r"electron antineutrino luminosity",
            unit=r"$10^{51}\,\mathrm{erg\,s}^{-1}$",
            code_unit=r"$10^{51} M_\odot M_\odot^{-1}$"
        ),
        reduction=integral,
        scale_factor=Units["Energy"]/Units["Time"]*1e3
    ),
    'L-x': dict(
        dependencies=('L-nu-x',),
        func=lambda dd, *_, **kw: dd,
        plot_name_kwargs=dict(
            name=r"'x' neutrino luminosity",
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
    'L-e-ns': dict(
        dependencies=('L-nu-e', 'dens', 'dens-ns'),
        func=lambda L, dd, dns, *_, **kw: L*(dd >= dns),
        plot_name_kwargs=dict(
            name=r"HMNS electron neutrino luminosity",
            unit=r"$10^{51}\,\mathrm{erg\,s}^{-1}$",
            code_unit=r"$10^{51} M_\odot M_\odot^{-1}$"
        ),
        reduction=integral,
        scale_factor=Units["Energy"]/Units["Time"]*1e3
    ),
    'L-a-ns': dict(
        dependencies=('L-nu-a', 'dens', 'dens-ns'),
        func=lambda L, dd, dns, *_, **kw: L*(dd >= dns),
        plot_name_kwargs=dict(
            name=r"HMNS electron antineutrino luminosity",
            unit=r"$10^{51}\,\mathrm{erg\,s}^{-1}$",
            code_unit=r"$10^{51} M_\odot M_\odot^{-1}$"
        ),
        reduction=integral,
        scale_factor=Units["Energy"]/Units["Time"]*1e3
    ),
    'L-x-ns': dict(
        dependencies=('L-nu-x', 'dens', 'dens-ns'),
        func=lambda L, dd, dns, *_, **kw: L*(dd >= dns),
        plot_name_kwargs=dict(
            name=r"HMNS 'x' neutrino luminosity",
            unit=r"$10^{51}\,\mathrm{erg\,s}^{-1}$",
            code_unit=r"$10^{51} M_\odot M_\odot^{-1}$"
        ),
        reduction=integral,
        scale_factor=Units["Energy"]/Units["Time"]*1e3
    ),
    'L-tot-ns': dict(
        dependencies=('L-e-ns', 'L-a-ns', 'L-x-ns'),
        func=lambda e, a, x, *_, **kw: e+a+x,
        plot_name_kwargs=dict(
            name=r"total HMNS neutrino luminosity",
            unit=r"$10^{51}\,\mathrm{erg\,s}^{-1}$",
            code_unit=r"$10^{51} M_\odot M_\odot^{-1}$"
        ),
        scale_factor=Units["Energy"]/Units["Time"]*1e3
    ),
    'L-e-disk': dict(
        dependencies=('L-nu-e', 'dens', 'dens-ns'),
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
        dependencies=('L-nu-a', 'dens', 'dens-ns'),
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
        dependencies=('L-nu-x', 'dens', 'dens-ns'),
        func=lambda L, dd, dns, *_, **kw: L*(dd < dns),
        plot_name_kwargs=dict(
            name=r"disk 'x' neutrino luminosity",
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
            unit="$M_\odot$ ms$^{-1}$",
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
            unit="$M_\odot$ ms$^{-1}$",
            code_unit="",
        ),
        reduction=sphere_surface_integral,
        scale_factor=RUnits['Time'],
    ),
    'dens-ns': dict(
        dependencies=("dens",),
        func=lambda *ar, **kw: 1,
        plot_name_kwargs=dict(
            name="density on HMNS surface",
            unit="g cm$^{-3}$",
            code_unit="$M_\odot^{-2}$"
        ),
        kwargs=dict(
            func='log'
        ),
        reduction=_dens_ns,
        scale_factor='Rho',
    ),
    'mass-ns': dict(
        dependencies=("dens", "dens-ns"),
        func=lambda dens, dens_ns, *_, **kw: 2*dens*(dens >= dens_ns).astype(int),
        plot_name_kwargs=dict(
            name="HMNS mass",
            unit="$M_\odot$"
        ),
        reduction=integral,
    ),
    'mass-disk': dict(
        dependencies=("baryon-mass", "mass-ns"),
        func=lambda mtot, mns, *_, **kw: mtot-mns,
        plot_name_kwargs=dict(
            name="disk mass",
            unit="$M_\odot$"
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
            unit="$M_\odot$ ms$^{-1}$",
            code_unit="",
            format_func=dict(
                radius=lambda radius, code_units:
                (f"{radius:.0f} " + '$M_\\odot$' if code_units else f"{radius*Units['Length']:.0f} km")
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
            unit="$M_\odot$",
            format_func=dict(
                radius=lambda radius, code_units:
                (f"{radius:.0f} " + '$M_\\odot$' if code_units else f"{radius*Units['Length']:.0f} km")
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
            unit="$M_\odot$",
            format_func=dict(
                radius=lambda radius, code_units:
                (f"{radius:.0f} " + '$M_\\odot$' if code_units else f"{radius*Units['Length']:.0f} km")
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
            unit="$M_\odot$",
            format_func=dict(
                radius=lambda radius, code_units:
                (f"{radius:.0f} " + '$M_\\odot$' if code_units else f"{radius*Units['Length']:.0f} km")
            ),
        ),
        save=False,
        PPkeys=['radius'],
    ),
}
