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


def _mass_flow_cartoon(vx, vz, bx, bz,
                       gxx, gxz, gzz,
                       alp, dens,
                       x, z, **_):
    Vx = alp*vx - bx
    Vz = alp*vz - bz
    r = (gxx*x**2 + gzz*z**2 + 2*gxz*x*z)**.5
    return dens*(gxx*Vx*x + gzz*Vz*z + gxz*(Vx*z + Vz*x))/r


user_variables = {
    'psi-max': dict(
        dependencies=('phi_max',),
        func=lambda phi, *_, **kw: np.exp(phi),
        plot_name_kwargs=dict(
            name="maximum conformal factor",
        ),
    ),
    'psi-min': dict(
        dependencies=('phi_min',),
        func=lambda phi, *_, **kw: np.exp(phi),
        plot_name_kwargs=dict(
            name="minimum conformal factor",
        ),
    ),
    'baryon-mass': dict(
        dependencies=('dens',),
        func=lambda dd, *_, **kw: dd,
        plot_name_kwargs=dict(
            name="total baryon mass",
            unit="$M_\odot$"
        ),
        reduction=integral
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
}
