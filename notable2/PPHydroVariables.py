import numpy as np
from scipy.interpolate import interpn
from scipy.optimize import minimize


from notable2.Utils import RUnits, Units


def _r_proj(vx, vy, vz,
            gxx, gxy, gxz,
            gyy, gyz, gzz,
            x=0, y=0, z=0, **_):
    x, y, z = [cc.squeeze() for cc in np.meshgrid(x, y, z, indexing='ij')]
    r = (gxx*x**2 + gyy*y**2 + gzz*z**2 +
         2*gxy*x*y + 2*gxz*x*z + 2*gyz*y*z)**.5
    vr = (gxx*vx*x + gyy*vy*y + gzz*vz*z +
          gxy*(vx*y + vy*x) +
          gxz*(vx*z + vz*x) +
          gyz*(vy*z + vz*y)
          )/r
    return vr


def _radial(vx, vy, vz, x=0, y=0, z=0, **_):
    # coords = [xyz for xyz in (x, y, z) if isinstance(xyz, np.ndarray)]
    # ori = minimize(lambda xy: interpn(coords, alp, xy, method='linear')[0],
    #               np.array([0. for _ in coords]))['x']
    # for ii, oo in enumerate(ori):
    #    coords[ii] -= oo
    x, y, z = [cc.squeeze() for cc in np.meshgrid(x, y, z, indexing='ij')]
    r = (x**2 + y**2 + z**2)**.5
    r[r == 0] = 1
    return (vx*x + vy*y + vz*z)/r


def _Omega(vx, vy, x=0, y=0, z=0, **_):
    x, y, z = [cc.squeeze() for cc in np.meshgrid(x, y, z, indexing='ij')]
    r = (x**2 + y**2)**.5
    r[r == 0] = np.inf
    return (x*vy - y*vx)/r**2


def _Omega_excised(vx, vy, x=0, y=0, z=0, **_):
    x, y, z = [cc.squeeze() for cc in np.meshgrid(x, y, z, indexing='ij')]
    r = (x**2 + y**2)**.5
    r[r <= 2] = np.inf
    return (x*vy - y*vx)/r**2


def _J_phi(dd, hh, ww,
           vx, vy, vz,
           gxx, gxy, gxz,
           gyy, gyz,
           x=0, y=0, z=0, **_):
    x, y, z = [cc.squeeze() for cc in np.meshgrid(x, y, z, indexing='ij')]
    # r = (gxx*x**2 + gyy*y**2 +
    #      2*gxy*x*y + gxz*x*z + gyz*y*z)**.5
    r = (x**2 + y**2)**.5
    r[r == 0] = np.inf
    v_y = gxy*vx + gyy*vy + gyz*vz
    v_x = gxx*vx + gxy*vx + gxz*vz
    return dd*hh*ww*(x*v_y - y*v_x)


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
    "h": dict(
        dependencies=('eps', 'press', 'rho'),
        func=lambda eps, press, rho, *_, **kw: 1 + eps + press/rho,
        plot_name_kwargs=dict(
            name="specific enthalpy",
        ),
        kwargs=dict(
            cmap='inferno',
        )
    ),
    "E-rel": dict(
        dependencies=('rho', 'h', 'W', 'press'),
        func=lambda rho, hh, ww, press, *_, **kw: rho*hh*ww**2 - press,
        plot_name_kwargs=dict(
            name="relativistic energy density",
        ),
        kwargs=dict(
            cmap='inferno',
        )
    ),
    "vel^r": dict(
        dependencies=('vel^x', 'vel^y', 'vel^z'),  # , 'alpha'),
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
    "J_phi": dict(
        dependencies=('dens', 'h', 'W',
                      'vel^x', 'vel^y', 'vel^z',
                      'g_xx', 'g_xy', 'g_xz',
                      'g_yy', 'g_yz'),
        func=_J_phi,
        plot_name_kwargs=dict(
            name="anular momentum density",
        ),
        kwargs=dict(
            cmap='seismic',
            symetric_around=0,
        )
    ),
    "V^x": dict(
        dependencies=('vel^x', 'beta^x', 'alpha'),
        func=lambda v, b, a, *_, **kw: a*v-b,
        plot_name_kwargs=dict(
            name="$V^x$",
            unit='$c$',
        ),
        kwargs=dict(
            cmap='seismic',
            symetric_around=0,
        ),
    ),
    "V^y": dict(
        dependencies=('vel^y', 'beta^y', 'alpha'),
        func=lambda v, b, a, *_, **kw: a*v-b,
        plot_name_kwargs=dict(
            name="$V^y$",
            unit='$c$',
        ),
        kwargs=dict(
            cmap='seismic',
            symetric_around=0,
        ),
    ),
    "V^z": dict(
        dependencies=('vel^z', 'beta^z', 'alpha'),
        func=lambda v, b, a, *_, **kw: a*v-b,
        plot_name_kwargs=dict(
            name="$V^z$",
            unit='$c$',
        ),
        kwargs=dict(
            cmap='seismic',
            symetric_around=0,
        ),
    ),
    "V^r": dict(
        dependencies=('V^x', 'V^y', 'V^z',
                      'g_xx', 'g_xy', 'g_xz',
                      'g_yy', 'g_yz', 'g_zz'),  # , 'alpha'),
        func=_r_proj,
        plot_name_kwargs=dict(
            name="$V^r$",
            unit='$c$',
        ),
        kwargs=dict(
            cmap='seismic',
            symetric_around=0,
        ),
    ),
    "radial-flow": dict(
        dependencies=('V^r', 'dens'),
        func=lambda vv, dd, *_, **kw: vv*dd,
        plot_name_kwargs=dict(
            name="$radial mass flow$",
            unit='$M_\\odot ^{-2}$',
        ),
        kwargs=dict(
            cmap='cubehelix',
        ),
        save=False,
    ),
    "ejb-flow": dict(
        dependencies=('V^r', 'dens', 'h', 'u_t'),
        func=lambda vv, dd, hh, ut, *_, **kw: vv*dd*(hh*ut < -1),
        plot_name_kwargs=dict(
            name="$ejecta flow$",
            unit='$M_\\odot ^{-2}$',
        ),
        kwargs=dict(
            cmap='cubehelix',
        )
    ),
    "ejg-flow": dict(
        dependencies=('V^r', 'dens', 'u_t'),
        func=lambda vv, dd, ut, *_, **kw: vv*dd*(ut < -1),
        plot_name_kwargs=dict(
            name="$ejecta flow$",
            unit='$M_\\odot ^{-2}$',
        ),
        kwargs=dict(
            cmap='cubehelix',
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
    ),
    "Omega": dict(
        dependencies=("V^x", "V^y"),
        func=_Omega,
        plot_name_kwargs=dict(
            name="$\Omega$",
            unit="rad ms$^{-1}$",
            code_unit="rad $M_\\odot^{-1}$",
        ),
        kwargs=dict(
            cmap="viridis",
        ),
        scale_factor=RUnits['Time']
    ),
    "Omega-excised": dict(
        dependencies=("V^x", "V^y"),
        func=_Omega_excised,
        plot_name_kwargs=dict(
            name="$\Omega$",
            unit="rad ms$^{-1}$",
            code_unit="rad $M_\\odot^{-1}$",
        ),
        kwargs=dict(
            cmap="viridis",
        ),
        scale_factor=RUnits['Time']
    ),
    "u_t-pp": dict(
        dependencies=('W', 'alpha',
                      'vel^x', 'vel^y', 'vel^z',
                      'g_xx', 'g_yy', 'g_zz',
                      'g_xy', 'g_xz', 'g_yz',
                      'beta^x', 'beta^y', 'beta^z',),
        func=lambda W, alp, vx, vy, vz,
        gxx, gyy, gzz, gxy, gxz, gyz,
        bx, by, bz, *_, **kw:
        W * (-alp + vx*bx*gxx + vy*by*gyy + vz*bz*gzz +
             (vx*by + bx*vy)*gxy +
             (vx*bz + bx*vz)*gxz +
             (vy*bz + by*vz)*gyz),
        plot_name_kwargs=dict(
            name=r"$u_t$",
            unit="$c$"
        ),
        kwargs=dict(
            symetric_around=-1,
            cmap='seismic'
        )
    ),
    "h-u_t": dict(
        dependencies=('h', 'u_t'),
        func=lambda h, ut, *_, **kw: h*ut,
        plot_name_kwargs=dict(
            name=r"$h u_t$",
            unit="$c$"
        ),
        save=False,
        kwargs=dict(
            symetric_around=-1,
            cmap='seismic'
        )
    ),
}
