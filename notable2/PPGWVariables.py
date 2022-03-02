from notable2.ExtractGW import extract_strain


pp_variables = {
    'h+': dict(
        func=extract_strain,
        plot_name_kwargs=dict(
            name=r"$h_+^{ll mm}$",
            format_opt=dict(
                ll=lambda ll, **_: str(ll),
                mm=lambda mm, **_: str(mm)
            )
        ),
        PPkeys=dict(ll=2, mm=2, power=1, n_points=3000, u_junk=200.)
    ),
    'hx': dict(
        func=extract_strain,
        plot_name_kwargs=dict(
            name=r"$h_\times^{ll mm}$",
            format_opt=dict(
                ll=lambda ll, **_: str(ll),
                mm=lambda mm, **_: str(mm)
            )
        ),
        PPkeys=dict(ll=2, mm=2, power=1, n_points=3000, u_junk=200.)
    ),
}
