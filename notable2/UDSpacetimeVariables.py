import numpy as np

user_variables = {
    "psi": dict(
        backups="psi-ud",
        dependencies=("phi",),
        func=lambda phi, *_, **kw: np.exp(phi),
        plot_name_kwargs=dict(
            name="conformal factor",
        ),
        kwargs=dict(
            cmap='plasma',
        )
    ),
    "phi-ud": dict(
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
    "psi-ud": dict(
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
}
