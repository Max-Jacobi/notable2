from notable2.EOS import EOS
from typing import Any

from notable2.Utils import RUnits


def pp_variables(eos: EOS) -> dict[str, dict[str, Any]]:
    ppvars = {
        'eos_h': dict(
            dependencies=['ye', 'temp', 'rho'],
            func=eos.get_caller(['internalEnergy', 'pressure'],
                                func=lambda eps, pres, rho, *_, **kw:
                                1 + RUnits["Eps"]*eps + RUnits["Press"]*pres/rho),
            plot_name_kwargs=dict(name="specific relativistic enthalpy"),
            kwargs=dict(cmap='hot')
        )
    }
    return ppvars
