import numpy as np
from notable2.EOS import EOS
from typing import Any

from notable2.Utils import Units, RUnits, Dict


def _Gamma(rho, eps, press, *_, **kw):
    gamma = press/eps/rho + 1
    gamma[rho <= 3e-15] = np.nan
    gamma[press <= 1e-10] = np.nan
    gamma[eps < 1e-10] = np.nan
    return gamma


def pp_variables(eos: EOS) -> Dict[str, Dict[str, Any]]:
    ppvars = {
        'ndens-eos': dict(
            dependencies=['rho'],
            func=lambda rho, *_, **__: rho/eos.get_mbary50(),
            plot_name_kwargs=dict(name="number density [fm$^{-3}$]"),
            kwargs=dict(cmap='viridis'),
            scale_factor=1/Units['Length']**3/1e4
        ),
        'h-eos': dict(
            dependencies=['ye', 'temp', 'rho'],
            func=eos.get_caller(['internalEnergy', 'pressure'],
                                func=lambda eps, pres, rho, *_, **kw:
                                1 + eps + pres/rho),
            plot_name_kwargs=dict(name="specific relativistic enthalpy"),
            kwargs=dict(cmap='hot'),
        ),
        'h-inf': dict(
            dependencies=['ye'],
            func=eos.get_inf_caller(['internalEnergy'],
                                    func=lambda eps, mfac, *_, **__:
                                    1 + eps),
            plot_name_kwargs=dict(name=r"$h_\infty$"),
            kwargs=dict(cmap='cubehelix'),
        ),
        'eps-eos': dict(
            dependencies=['ye', 'temp', 'rho'],
            func=eos.get_caller(['internalEnergy'],
                                func=lambda eps, *_, **kw: eps),
            plot_name_kwargs=dict(name="specific internal energy"),
            kwargs=dict(cmap='inferno'),
        ),
        'eps-eos-baryonic': dict(
            dependencies=['ye', 'temp', 'rho'],
            func=eos.get_caller(['only_E'],
                                func=lambda eps, *_, **kw: eps),
            plot_name_kwargs=dict(
                name="specific internal energy (baryonic only)"),
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
        'press-eos-baryonic': dict(
            dependencies=['ye', 'temp', 'rho'],
            func=eos.get_caller(['only_P'],
                                func=lambda pres, *_, **kw: pres),
            plot_name_kwargs=dict(
                name=r"pressure (baryonic only)",
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
        'press-cold-eos-baryonic': dict(
            dependencies=['ye', 'rho'],
            func=eos.get_cold_caller(['only_P'],
                                     func=lambda pres, *_, **kw: pres),
            plot_name_kwargs=dict(
                name=r"pressure ($T=0$) (baryonic only)",
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
        'eps-cold-eos-baryonic': dict(
            dependencies=['ye', 'rho'],
            func=eos.get_cold_caller(['only_E'],
                                     func=lambda eps, *_, **kw: eps),
            plot_name_kwargs=dict(
                name="specific internal energy ($T=0$) (baryonic only)"),
            kwargs=dict(cmap='inferno'),
        ),
        'press-th-eos': dict(
            dependencies=['press-eos', 'press-cold-eos'],
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
        'press-th-eos-baryonic': dict(
            dependencies=['press-eos-baryonic', 'press-cold-eos-baryonic'],
            func=lambda press, pressc, *_, **kw: press-pressc,
            save=False,
            plot_name_kwargs=dict(
                name=r"thermal pressure (baryonic only)",
                code_unit="$M_\\odot^{-2}$",
                unit="g cm$^{-1}$ s$^{-2}$",
            ),
            kwargs=dict(cmap='plasma'),
            scale_factor="Press"
        ),
        'press-th/tot-eos': dict(
            dependencies=['press-eos', 'press-cold-eos'],
            func=lambda press, pressc, *_, **kw: (press-pressc)/press,
            save=False,
            plot_name_kwargs=dict(
                name=r"$P_{\rm th}/P_{\rm tot}$",
            ),
            kwargs=dict(cmap='plasma', func='log'),
        ),
        'press-th/cold-eos': dict(
            dependencies=['press-eos', 'press-cold-eos'],
            func=lambda press, pressc, *_, **kw: (press-pressc)/pressc,
            save=False,
            plot_name_kwargs=dict(
                name=r"$P_{\rm th}/P_{\rm cold}$",
            ),
            kwargs=dict(cmap='plasma', func='log'),
        ),
        'eps-th-eos': dict(
            dependencies=['eps-eos', 'eps-cold-eos'],
            save=False,
            func=lambda eps, epsc, *_, **kw: eps-epsc,
            plot_name_kwargs=dict(name=r"$\epsilon_{\rm th}$"),
            kwargs=dict(cmap='inferno'),
        ),
        'eps-th-eos-baryonic': dict(
            dependencies=['eps-eos-baryonic', 'eps-cold-eos-baryonic'],
            save=False,
            func=lambda eps, epsc, *_, **kw: eps-epsc,
            plot_name_kwargs=dict(name=r"$\epsilon_{\rm th}$ (baryonic only)"),
            kwargs=dict(cmap='inferno'),
        ),
        'e-th-eos': dict(
            dependencies=['eps-th-eos', 'press-th-eos', 'W', 'rho', 'psi'],
            save=False,
            func=lambda eps, press, Wl, rho, psi, *
            _, **kw: psi**6 * (Wl**2 * (eps*rho + press) - press),
            plot_name_kwargs=dict(name=r"$e_{\rm th}$"),
            kwargs=dict(cmap='inferno'),
            scale_factor="Rho"
        ),
        'Gamma-th': dict(
            dependencies=['rho', 'eps-th-eos', 'press-th-eos'],
            save=False,
            func=_Gamma,
            plot_name_kwargs=dict(name=r"$\Gamma_{\rm th}$"),
            kwargs=dict(cmap='cubehelix'),
        ),
        'Gamma-th-baryonic': dict(
            dependencies=['rho', 'eps-th-eos-baryonic',
                          'press-th-eos-baryonic'],
            save=False,
            func=_Gamma,
            plot_name_kwargs=dict(name=r"$\Gamma_{\rm th}$ (baryonic only)"),
            kwargs=dict(cmap='cubehelix'),
        ),
    }
    return ppvars
