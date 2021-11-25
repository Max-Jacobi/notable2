from notable2.EOS import EOS
from typing import Any

from notable2.Utils import RUnits


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
    }
    return ppvars
