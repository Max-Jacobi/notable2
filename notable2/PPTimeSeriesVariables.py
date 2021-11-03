import numpy as np
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


def _times_domain_volume(dependencies, its, sim, func, **kwargs):
    bmass = dependencies[0].get_data(it=its)
    coords = sim.get_coords('xyz', 0, exclude_ghosts=3)[0]
    vol = np.prod([cc[-1] - cc[0] for cc in coords.values()])
    return func(bmass.data)*vol


def _dens_ns(dependencies, its, sim, **kwargs):
    region = 'xz' if sim.is_cartoon else 'xyz'
    rl = sim.rls.max()
    result = np.zeros_like(its, dtype=float)

    for ii, it in enumerate(its):
        dens = dependencies[0].get_data(region=region, it=it)
        dds = dens[rl]
        coords = dens.coords[rl]

        vols = np.ones_like(dds)
        if sim.is_cartoon:
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
    'L-tot': dict(
        dependencies=('L-nu-tot',),
        func=lambda dd, *_, **kw: dd,
        plot_name_kwargs=dict(
            name=r"total neutrino luminosity",
            unit=r"$10^{51}\,\mathrm{erg\,s}^{-1}$",
            code_unit=r"$10^{51} M_\odot M_\odot^{-1}$"
        ),
        reduction=integral,
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
        dependencies=("dens", "dens-ns"),
        func=lambda dens, dens_ns, *_, **kw: 2*dens*(dens < dens_ns).astype(int),
        plot_name_kwargs=dict(
            name="disk mass",
            unit="$M_\odot$"
        ),
        reduction=integral,
    ),
    'ejecta-mass': dict(
        dependencies=("vel^x", "vel^y", "vel^z",
                      "beta^x", "beta^y", "beta^z",
                      'g_xx', 'g_xy', 'g_xz',
                      'g_yy', 'g_yz', 'g_zz',
                      "alpha", "dens", "u_t"),
        func=_mass_flow_ej,
        plot_name_kwargs=dict(
            name="mass flux",
            unit="$M_\odot$ ms$^{-1}$",
            code_unit="",
        ),
        reduction=sphere_surface_integral,
        scale_factor=1/Units['Time'],
    ),
}
