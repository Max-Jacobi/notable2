import numpy as np


def _radial(vx, vy, vz, x=0, y=0, z=0, **_):
    x, y, z = [cc.squeeze() for cc in np.meshgrid(x, y, z, indexing='ij')]
    r = (x**2 + y**2 + z**2)**.5
    r[r == 0] = 1
    return (vx*x + vy*y + vz*z)/r


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
    "vel^r": dict(
        dependencies=('vel^x', 'vel^y', 'vel^z'),
        func=_radial,
        plot_name_kwargs=dict(
            name="radial velocity",
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
        func=lambda vx, vy, vz, gxx, gxy, gxz, gyy, gyz, gzz, *_, **kw: 1./np.sqrt(
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
            name="$L_{\\nu_{\\rm tot}}$",
        ),
        kwargs=dict(
            cmap="viridis",
            func="log"
        ),
    )
}
