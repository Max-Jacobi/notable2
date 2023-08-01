import numpy as np
from typing import Dict, Any


from notable2.Utils import RUnits, Units
from notable2.EOS import EOS


def _tau_exp(vr, x=0, y=0, z=0, **_):
    mm = vr <= 0
    vr[mm] = 1
    ret = np.sqrt(_r2(x, y, z))/vr
    ret[mm] = np.nan
    return ret


def _r2(x=0, y=0, z=0):
    x, y, z = [cc.squeeze() for cc in np.meshgrid(x, y, z, indexing='ij')]
    return x**2 + y**2 + z**2


def _ej_dens(dd, hut, inner_r=300, outer_r=1000, x=0, y=0, z=0):
    res = dd*(hut < -1)
    x, y, z = [cc.squeeze() for cc in np.meshgrid(x, y, z, indexing='ij')]
    r2 = x**2 + y**2 + z**2
    mask = np.logical_or(r2 > outer_r**2, r2 < inner_r**2)
    res[mask] = 0
    return res


def _vinf(ut, *_, **__):
    mask = ut < -1
    res = np.zeros_like(ut)
    res[mask] = (1 - ut[mask]**-2)**.5
    return res


def _r_proj(vx, vy, vz,
            gxx, gxy, gxz,
            gyy, gyz, gzz,
            x=0, y=0, z=0, **_):
    x, y, z = [cc.squeeze() for cc in np.meshgrid(x, y, z, indexing='ij')]
    rr = (x**2 + y**2 + z**2)**.5
    vr = (gxx*vx*x + gyy*vy*y + gzz*vz*z +
          gxy*(vx*y + vy*x) +
          gxz*(vx*z + vz*x) +
          gyz*(vy*z + vz*y)
          )/rr
    if np.any(vr > 1):
        breakpoint()
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
    om = (x*vy - y*vx)/r**2
    om[om > 300] = np.nan
    return om


def _j_phi(hh, ww,
           vx, vy, vz,
           gxx, gxy, gxz,
           gyy, gyz,
           x=0, y=0, z=0, **_):
    x, y, z = [cc.squeeze() for cc in np.meshgrid(x, y, z, indexing='ij')]
    u_y = (gxy*vx + gyy*vy + gyz*vz)*ww
    u_x = (gxx*vx + gxy*vx + gxz*vz)*ww
    return hh*(x*u_y - y*u_x)


def _radius_format_func(code_units=False, **kwargs):
    for name in ['radius', 'inner_r', 'outer_r']:
        if name in kwargs:
            radius = kwargs[name]
            break
    if code_units:
        return f"{radius:.0f} " + r'M_\odot$'
    else:
        return f"{radius*Units['Length']:.0e}"+r"\,$km"


def pp_variables(eos: EOS) -> Dict[str, Dict[str, Any]]:
    ppvars = {
        "dens-pp": dict(
            dependencies=('rho', 'W', 'vol-fac'),
            func=lambda W, rho, sqg, *_, **kw: rho*W*sqg,
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
        "E-con": dict(
            dependencies=('rho', 'h', 'W', 'press'),
            func=lambda rho, hh, ww, press, *_, **kw: rho*hh*ww**2 - press,
            plot_name_kwargs=dict(
                name="relativistic energy density",
            ),
            kwargs=dict(
                cmap='inferno',
            )
        ),
        "tau-con": dict(
            dependencies=('rho', 'eps', 'W', 'press', 'phi'),
            func=lambda rho, eps, ww, press, phi, *_, **kw:
            phi**-3*(rho*ww*(eps*ww+ww-1) + press*(ww*ww-1)),
            plot_name_kwargs=dict(
                name="conserverd internal energy density",
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
        "j_phi": dict(
            dependencies=('h', 'W',
                          'vel^x', 'vel^y', 'vel^z',
                          'g_xx', 'g_xy', 'g_xz',
                          'g_yy', 'g_yz'),
            func=_j_phi,
            plot_name_kwargs=dict(
                name="specific angular momentum density",
            ),
            kwargs=dict(
                cmap='cubehelix',
            )
        ),
        "J_phi": dict(
            dependencies=('dens', 'j_phi'),
            func=lambda dd, jj, *_, **__: dd*jj,
            save=False,
            plot_name_kwargs=dict(
                name="angular momentum density",
            ),
            kwargs=dict(
                cmap='cubehelix',
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
            dependencies=('V^r', 'dens', 'h-u_t'),
            func=lambda vv, dd, hut, *_, **kw: vv*dd*(hut < -1),
            plot_name_kwargs=dict(
                name="ejecta flow",
                unit='$M_\\odot ^{-2}$',
            ),
            kwargs=dict(
                cmap='cubehelix',
                func='log',
            )
        ),
        "ejg-flow": dict(
            dependencies=('V^r', 'dens', 'u_t'),
            func=lambda vv, dd, ut, *_, **kw: vv*dd*(ut < -1),
            plot_name_kwargs=dict(
                name="ejecta flow",
                unit='$M_\\odot ^{-2}$',
            ),
            kwargs=dict(
                cmap='cubehelix',
                func='log',
            )
        ),
        "ejb-dens": dict(
            dependencies=('dens', 'h-u_t'),
            func=_ej_dens,
            plot_name_kwargs=dict(
                name=r"ejecta density ($inner_r \leq r \leq outer_r)",
                unit='$M_\\odot ^{-2}$',
                format_opt=dict(
                    inner_r=_radius_format_func,
                    outer_r=_radius_format_func
                ),
            ),
            kwargs=dict(
                cmap='cubehelix',
                func='log',
            ),
            PPkeys=dict(inner_r=300, outer_r=1000),
        ),
        "ejg-dens": dict(
            dependencies=('dens', 'u_t'),
            func=_ej_dens,
            plot_name_kwargs=dict(
                name=r"ejecta density ($inner_ \leq r \leq outer_r)",
                unit='$M_\\odot ^{-2}$',
                format_opt=dict(
                    inner_r=_radius_format_func,
                    outer_r=_radius_format_func
                ),
            ),
            kwargs=dict(
                cmap='cubehelix',
                func='log',
            ),
            PPkeys=dict(inner_r=300, outer_r=1000),
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
        "tau-exp": dict(
            dependencies=("vel^r",),
            func=_tau_exp,
            save=False,
            scale_factor='Time',
            plot_name_kwargs=dict(
                name=r"$\tau_{\rm exp}$",
                unit="ms",
                code_unit="$M_\\odot$",
            ),
            kwargs=dict(
                cmap="cubehelix_r",
                func="log"
            ),
        ),
        "tau-nu-em": dict(
            dependencies=("ndens-eos", "ye", "R-nu-e",
                          "R-nu-a", "alpha", "vol-fac"),
            func=lambda nn, ye, rnue, rnua, alp, vf, *_, **__:
            nn/(rnua - rnue)/alp/vf,
            save=False,
            plot_name_kwargs=dict(
                name=r"$\tau_{\nu{\rm ,em.}}$",
                unit="ms",
                code_unit="$M_\\odot$",
            ),
            scale_factor="Time",
            kwargs=dict(
                cmap="cubehelix_r",
                func="log"
            ),
        ),
        "tau-nu-em-en": dict(
            dependencies=("rho", "eps", "Q-nu-e", "Q-nu-a", "Q-nu-x"),
            func=lambda rho, eps, qe, qa, qx, *_, **__: rho*eps/(qe+qa+qx),
            save=False,
            plot_name_kwargs=dict(
                name=r"$\tau_{\nu{\rm ,em., en.}}$",
                unit="ms",
                code_unit="$M_\\odot$",
            ),
            scale_factor="Time",
            kwargs=dict(
                cmap="cubehelix_r",
                func="log"
            ),
        ),
        "tau-nu-abs": dict(
            dependencies=("rho", "ye", "nu-abs-num", "alpha", 'vol-fac'),
            func=lambda rho, ye, rnu, alp, vf, * \
                _, **__: rho/rnu/alp/vf,
            save=False,
            plot_name_kwargs=dict(
                name=r"$\tau_{\nu{\rm ,abs.}}$",
                unit="ms",
                code_unit="$M_\\odot$",
            ),
            scale_factor="Time",
            kwargs=dict(
                cmap="cubehelix_r",
                func="log"
            ),
        ),
        "tau-nu-abs-en": dict(
            dependencies=("rho", "eps", "nu-abs-en",),
            func=lambda rho, eps, rnu, *_, **__: rho*eps/rnu,
            save=False,
            plot_name_kwargs=dict(
                name=r"$\tau_{\nu{\rm ,abs., en.}}$",
                unit="ms",
                code_unit="$M_\\odot$",
            ),
            scale_factor="Time",
            kwargs=dict(
                cmap="cubehelix_r",
                func="log"
            ),
        ),
        "Q-nu-tot": dict(
            dependencies=("Q-nu-e", "Q-nu-a", "Q-nu-x",
                          "nu-abs-en"),
            func=lambda qe, qa, qx, qabs, *_, **__: qabs - qe - qa - qx,
            save=False,
            plot_name_kwargs=dict(
                name=r"$q_{\nu}$",
            ),
            kwargs=dict(
                cmap="seismic",
                symetric_around=0,
            ),
        ),
        "R-nu-tot": dict(
            dependencies=("R-nu-e", "R-nu-a", "nu-abs-num"),
            func=lambda re, ra, rabs, *_, **__:
            rabs - eos.get_mbary50()*(re - ra),
            save=False,
            plot_name_kwargs=dict(
                name=r"$r_{\nu}$",
            ),
            kwargs=dict(
                cmap="seismic",
                symetric_around=0,
            ),
        ),
        "tau-nu-tot": dict(
            dependencies=("R-nu-tot", "dens", "alpha", "vol-fac"),
            func=lambda rnu, dd, alp, vf, *_, **__: dd/(rnu*alp*vf),
            save=False,
            plot_name_kwargs=dict(
                name=r"$\tau_{\nu,{\rm tot}}$",
                unit="ms",
                code_unit="$M_\odot$",
            ),
            scale_factor="Time",
            kwargs=dict(
                cmap="seismic",
                symetric_around=0,
            ),
        ),
        "tau-nu-tot-en": dict(
            dependencies=("Q-nu-tot", "tau-con", "W", "alpha", "vol-fac"),
            func=lambda qnu, tau, ww, alp, vf, *_, **__:
            tau/(qnu*alp*vf*ww),
            save=False,
            plot_name_kwargs=dict(
                name=r"$\tau_{\nu, {\rm tot, en}}$",
                unit="ms",
                code_unit="$M_\odot$",
            ),
            scale_factor="Time",
            kwargs=dict(
                cmap="seismic",
                symetric_around=0,
            ),
        ),
        "tau-nu-em/abs": dict(
            dependencies=("tau-nu-em", "tau-nu-abs",),
            func=lambda taue, taua, *_, **__: taue/taua,
            save=False,
            plot_name_kwargs=dict(
                name=r"$\tau_{\nu{\rm ,em.}} / \tau_{\nu{\rm ,abs.}}$",
            ),
            kwargs=dict(
                cmap="seismic",
                func='log',
                symetric_around=1,
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
            dependencies=('h', 'h-inf', 'u_t'),
            func=lambda h, hinf, ut, *_, **kw: h/hinf*ut,
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
        "vel-inf-g": dict(
            dependencies=('u_t',),
            func=_vinf,
            plot_name_kwargs=dict(
                name=r"$v_\infty$",
                unit="$c$"
            ),
            save=False,
            kwargs=dict(
                cmap='cubehelix'
            )
        ),
        "vel-inf-b": dict(
            dependencies=('h-u_t',),
            func=_vinf,
            plot_name_kwargs=dict(
                name=r"$v_\infty$",
                unit="$c$"
            ),
            save=False,
            kwargs=dict(
                cmap='cubehelix'
            )
        ),
        'press-Gamma-th': dict(
            dependencies=("press-cold-eos", "rho", "eps-th-eos"),
            func=lambda pc, rho, epsth, gamma_th=5/3,
            *_, **__: pc + rho*epsth*(gamma_th-1),
            plot_name_kwargs=dict(
                name=r"pressure ($\Gamma_{\rm th} = gamma_th$)",
                format_opt=dict(
                    gamma_th=lambda gamma_th, **_: "{:.2f}".format(gamma_th),
                ),
                code_unit="$M_\\odot^{-2}$",
                unit="g cm$^{-1}$ s$^{-2}$",
            ),
            kwargs=dict(cmap='plasma', func='log'),
            scale_factor="Press",
            PPkeys=dict(gamma_th=5/3),
        ),
        'press-th-Gamma-th': dict(
            dependencies=("rho", "eps-th-eos"),
            func=lambda rho, epsth, gamma_th=5/3,
            *_, **__: rho*epsth*(gamma_th-1),
            plot_name_kwargs=dict(
                name=r"thermal pressure ($\Gamma_{\rm th} = gamma_th$)",
                format_opt=dict(
                    gamma_th=lambda gamma_th, **_: "{:.2f}".format(gamma_th),
                ),
                code_unit="$M_\\odot^{-2}$",
                unit="g cm$^{-1}$ s$^{-2}$",
            ),
            kwargs=dict(cmap='plasma', func='log'),
            scale_factor="Press",
            PPkeys=dict(gamma_th=5/3),
        ),
        'press-Gamma-th-ratio': dict(
            dependencies=('press', 'press-Gamma-th'),
            func=lambda p1, p2, *_, **__: p1/p2,
            plot_name_kwargs=dict(
                name=r"pressure ratio ($\Gamma_{\rm th} = gamma_th$)",
                format_opt=dict(
                    gamma_th=lambda gamma_th, **_: "{:.2f}".format(gamma_th),
                ),
            ),
            kwargs=dict(cmap='seismic', symetric_around=1),
            PPkeys=dict(gamma_th=5/3),
        ),
        'press-ideal': dict(
            dependencies=('rho', 'eps'),
            func=lambda rho, eps, *_, gamma=2, **__: (gamma - 1) * rho * eps,
            plot_name_kwargs=dict(
                name=r"pressure ",
                code_unit="$M_\\odot^{-2}$",
                unit="g cm$^{-1}$ s$^{-2}$",
            ),
            kwargs=dict(cmap='plasma', func='log'),
            scale_factor="Press",
            PPkeys=dict(gamma=2),
        ),
        'h-ideal': dict(
            dependencies=('rho', 'eps'),
            func=lambda rho, eps, *_, gamma=2, **__: 1 + gamma * eps,
            plot_name_kwargs=dict(
                name="specific enthalpy",
            ),
            kwargs=dict(
                cmap='inferno',
            )
        ),
    }
    return ppvars
