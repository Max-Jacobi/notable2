import numpy as np

user_variables = {
    "dens-ud": dict(
        dependencies=('rho', 'W', 'phi'),
        func=lambda W, rho, phi, *_, **kw: rho*W*np.exp(phi)**6,
        plot_name_kwargs=dict(
            name="conserved density",
            unit="g cm$^{-3}$",
            code_unit="$c M_\\odot^{-2}$",
        ),
        scale_factor="Rho",
        kwargs=dict(
            cmap='magma',
            func='log',
        )
    ),
    "vel^r-xz": dict(
        dependencies=('phi',),
        func=lambda phi, *_, **kw: np.exp(phi),
        plot_name_kwargs=dict(
            name="conformal factor",
            unit='$c$',
        ),
        kwargs=dict(
            cmap='seismic',
            symetric_around=0,
        )
    ),
    "W-ud": dict(
        dependencies=('vel^x', 'vel^y', 'vel^z',
                      'g_xx', 'g_xy', 'g_xz',
                      'g_yy', 'g_yz', 'g_zz',
                      ),
        func=lambda vx, vy, vz, gxx, gyy, gzz, gxy, gxz, gyz, *_, **kw: 1./np.sqrt(
            1. - (vx**2*gxx + vy**2*gyy + vz**2*gzz +
                  2.*(vx*vy*gxy + vx*vz*gxz + vy*vz*gyz))),
        plot_name_kwargs=dict(
            name="Lorentz factor",
        ),
        kwargs=dict(
            cmap='plasma',
        )
    ),
}
