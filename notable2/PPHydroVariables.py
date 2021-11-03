import numpy as np

pp_variables = {
    "dens-pp": dict(
        dependencies=('rho', 'W', 'phi'),
        func=lambda W, rho, phi, *_, **kw: rho*W*phi**-3,
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
    "W-pp": dict(
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
    "L-nu-tot": dict(
        dependencies=("L-nu-e", "L-nu-a", "L-nu-x"),
        func=lambda Le, La, Lx, *_, **kw: Le + La + Lx,
        plot_name_kwargs=dict(
            name="$L_{\\nu_{\\rm e}}$",
        ),
        kwargs=dict(
            cmap="viridis",
            func="log"
        ),
    )
}
