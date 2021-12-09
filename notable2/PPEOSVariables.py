from notable2.EOS import EOS
from typing import Any

from notable2.Utils import Units, RUnits


def pp_variables(eos: EOS) -> dict[str, dict[str, Any]]:
    ppvars = {
        'h-eos': dict(
            dependencies=['ye', 'temp', 'rho'],
            func=eos.get_caller(['internalEnergy', 'pressure'],
                                func=lambda eps, pres, rho, *_, **kw:
                                1 + eps + pres/rho),
            plot_name_kwargs=dict(name="specific relativistic enthalpy"),
            kwargs=dict(cmap='hot'),
        ),
        'eps-eos': dict(
            dependencies=['ye', 'temp', 'rho'],
            func=eos.get_caller(['internalEnergy'],
                                func=lambda eps, *_, **kw: eps),
            plot_name_kwargs=dict(name="specific internal energy"),
            kwargs=dict(cmap='inferno'),
        ),
        'entr-eos': dict(
            dependencies=['ye', 'temp', 'rho'],
            func=eos.get_caller(['entropy'],
                                func=lambda entr, *_, **kw: entr),
            plot_name_kwargs=dict(
                name=r"entropy",
                unit=r"$k_{\rm B}$/nuc.",
            ),
            kwargs=dict(cmap='inferno'),
        ),
        'press-eos': dict(
            dependencies=['ye', 'temp', 'rho'],
            func=eos.get_caller(['pressure'],
                                func=lambda pres, *_, **kw: pres),
            plot_name_kwargs=dict(
                name=r"pressure",
                code_unit="$M_\\odot^{-2}$",
                unit="g cm$^{-1}$ s$^{-2}$",
            ),
            kwargs=dict(cmap='plasma'),
            scale_factor="Press"
        ),
        'press-cold-eos': dict(
            dependencies=['ye', 'rho'],
            func=eos.get_cold_caller(['pressure'],
                                     func=lambda pres, *_, **kw: pres),
            plot_name_kwargs=dict(
                name=r"pressure ($T=0$)",
                code_unit="$M_\\odot^{-2}$",
                unit="g cm$^{-1}$ s$^{-2}$",
            ),
            kwargs=dict(cmap='plasma'),
            scale_factor="Press"
        ),
        'eps-cold-eos': dict(
            dependencies=['ye', 'rho'],
            func=eos.get_cold_caller(['internalEnergy'],
                                     func=lambda eps, *_, **kw: eps),
            plot_name_kwargs=dict(name="specific internal energy ($T=0$)"),
            kwargs=dict(cmap='inferno'),
        ),
        'press-th-eos': dict(
            dependencies=['press', 'press-cold-eos'],
            func=lambda press, pressc, *_, **kw: press-pressc,
            save=False,
            plot_name_kwargs=dict(
                name=r"thermal pressure",
                code_unit="$M_\\odot^{-2}$",
                unit="g cm$^{-1}$ s$^{-2}$",
            ),
            kwargs=dict(cmap='plasma'),
            scale_factor="Press"
        ),
        'eps-th-eos': dict(
            dependencies=['eps', 'eps-cold-eos'],
            save=False,
            func=lambda eps, epsc, *_, **kw: eps-epsc,
            plot_name_kwargs=dict(name=r"$\epsilon_{\rm th}$"),
            kwargs=dict(cmap='inferno'),
        ),
        'e-th-eos': dict(
            dependencies=['eps-th-eos', 'press-th-eos', 'W', 'rho', 'psi'],
            save=False,
            func=lambda eps, press, Wl, rho, psi, *_, **kw: psi**6 * (Wl**2 * (eps*rho + press) - press),
            plot_name_kwargs=dict(name=r"$e_{\rm th}$"),
            kwargs=dict(cmap='inferno'),
            scale_factor="Rho"
        ),
        'Gamma-th': dict(
            dependencies=['rho', 'eps-th-eos', 'press-th-eos'],
            func=lambda rho, eps, press, *_, **kw: press/eps/rho + 1,
            plot_name_kwargs=dict(name=r"$\Gamma_{\rm th}$"),
            kwargs=dict(cmap='cubehelix'),
        ),
    }
    return ppvars
