import numpy as np

pp_variables = {
    "psi": dict(
        backups=["psi-pp", "psi-BSSN"],
        dependencies=("phi",),
        func=lambda phi, *_, **kw: phi**-.5,
        plot_name_kwargs=dict(
            name="conformal factor",
        ),
        kwargs=dict(
            cmap='plasma',
        )
    ),
    "psi-BSSN": dict(
        dependencies=("phi-BSSN",),
        func=lambda phi, *_, **kw: np.exp("phi"),
        plot_name_kwargs=dict(
            name="conformal factor",
        ),
        kwargs=dict(
            cmap='plasma',
        )
    ),
    "phi-pp": dict(
        dependencies=("g_xx", "g_xy", "g_xz", "g_yy", "g_yz", "g_zz"),
        func=lambda xx, xy, xz, yy, yz, zz, *_, **kw:
        np.log10(xx*yy*zz + 2.*xy*yz*xz - xz**2*yy - xy**2*zz - yz**2*xx)/12,
        plot_name_kwargs=dict(
            name="log($phi$)",
        ),
        kwargs=dict(
            cmap="plasma",
        )
    ),
    "psi-pp": dict(
        dependencies=("g_xx", "g_xy", "g_xz", "g_yy", "g_yz", "g_zz"),
        func=lambda xx, xy, xz, yy, yz, zz, *_, **kw:
        (xx*yy*zz + 2.*xy*yz*xz - xz**2*yy - xy**2*zz - yz**2*xx)**(1/12),
        plot_name_kwargs=dict(
            name="conformal factor",
        ),
        kwargs=dict(
            cmap="plasma",
        )
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
}
