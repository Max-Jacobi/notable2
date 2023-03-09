import numpy as np

pp_variables = {
    "psi": dict(
        dependencies=("phi",),
        func=lambda phi, *_, **kw: np.exp(phi),
        plot_name_kwargs=dict(
            name="conformal factor",
        ),
        kwargs=dict(
            cmap='plasma',
        ),
        save=False
    ),
    "phi-pp": dict(
        dependencies=("vol-fac-pp",),
        func=lambda sqgam, *_, **kw: np.log(sqgam)/6,
        plot_name_kwargs=dict(
            name="$\phi$",
        ),
        kwargs=dict(
            cmap="plasma",
        ),
        save=False
    ),
    "vol-fac": dict(
        dependencies=("phi",),
        func=lambda phi, *_, **kw: np.exp(phi)**6,
        plot_name_kwargs=dict(
            name="conformal factor",
        ),
        kwargs=dict(
            cmap='plasma',
        ),
        save=False
    ),

    "vol-fac-pp": dict(
        dependencies=("g_xx", "g_xy", "g_xz", "g_yy", "g_yz", "g_zz"),
        func=lambda xx, xy, xz, yy, yz, zz, *_, **kw:
        (xx*yy*zz + 2.*xy*yz*xz - xz**2*yy - xy**2*zz - yz**2*xx)**0.5,
        plot_name_kwargs=dict(
            name=r"$\sqrt{\gamma}$",
        ),
        kwargs=dict(
            cmap="plasma",
        ),
        save=False
    ),
}
