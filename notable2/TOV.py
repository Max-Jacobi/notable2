import warnings
from typing import TYPE_CHECKING, Dict, Optional, Callable
import numpy as np
from scipy.integrate import solve_ivp, cumulative_trapezoid
from scipy.integrate._ivp.ivp import OdeResult
from scipy.interpolate import interp1d
from scipy.optimize import bisect, minimize_scalar
from .Utils import RUnits, Units
from .EOS import TabulatedEOS

if TYPE_CHECKING:
    from numpy.typing import NDArray

fourpi = 4 * np.pi
_lin_interp_keys = ['enthalpy', 'mu_nu', 'electron_fraction',
                    'Gamma', 'cs2', 'energy_density', ]


class TOVSolver:
    """
    A class for solving the Tolman-Oppenheimer-Volkoff (TOV) equations for a given equation of state (EOS).

    Parameters:
        equation_of_state (TabulatedEOS): The equation of state.
        verbose (bool, optional): If True, print progress and information during solving. Default is False.
        correct_pressure (bool, optional): If True, correct any non-monotonic pressure in the EOS table. Default is True.

    Attributes:
        parameters (dict): A dictionary of the TOV paramters obtained from solving.
        data (dict): A dictionary of the radial arrays obtained from solving.
        eos_table (dict): The one-dimensional eos table used for solving.
    """

    eos_table: Dict[str, "NDArray[np.float_]"]
    pizza_table: Dict[str, "NDArray[np.float_]"]
    data: Dict[str, "NDArray[np.float_]"]
    parameters: Dict[str, float]
    is_solved: bool = False

    def __init__(
        self,
        eos: TabulatedEOS,
        verbose: bool = False,
        correct_pressure: bool = True,
    ):
        """
        Initializes the TOV solver with the given equation of state.

        Args:
            equation_of_state (TabulatedEOS): The equation of state.
            verbose (bool, optional): If True, print progress and information during solving. Default is False.
            correct_pressure (bool, optional): If True, correct any non-monotonic pressure in the EOS table. Default is True.
        """
        self.parameters = {}
        self.pizza_table = {}
        self.data = {}
        self.eos = eos
        self.verbose = verbose
        self._generate_eos_table()
        if correct_pressure:
            self._correct_pressure()

    def _generate_eos_table(self):
        """
        Generate the EOS table required for TOV solving.
        """
        self.eos_table = {}
        self.eos_table["rho"] = self.eos.get_key("density")
        self.eos_table["ye"] = np.zeros_like(self.eos_table["rho"])
        ye_0 = self.eos.get_key("ye")
        rho, ye = np.meshgrid(self.eos_table["rho"], ye_0, indexing="ij")
        getter = self.eos.get_cold_caller(
            ["mu_p", "mu_e", "mu_n"],
            lambda mu_p, mu_e, mu_n, *_: mu_p + mu_e - mu_n
        )
        mu_nu = getter(ye, rho)

        for ii, (dd, mn) in enumerate(zip(rho, mu_nu)):
            f = interp1d(ye_0, mn, kind="linear", bounds_error=True)
            try:
                self.eos_table["ye"][ii] = bisect(
                    f, a=ye_0[0], b=ye_0[-1],
                    xtol=1e-6, rtol=1e-3, maxiter=100
                )
            except ValueError:
                self.eos_table["ye"][ii] = ye_0[0]

        self.eos_table["mu_nu"] = getter(
            self.eos_table["ye"], self.eos_table["rho"])
        getter = self.eos.get_cold_caller(["pressure"])
        self.eos_table["press"] = getter(
            self.eos_table["ye"], self.eos_table["rho"])
        getter = self.eos.get_cold_caller(["internalEnergy"])
        self.eos_table["eps"] = getter(
            self.eos_table["ye"], self.eos_table["rho"])
        getter = self.eos.get_cold_caller(["cs2"])
        self.eos_table["cs2"] = getter(
            self.eos_table["ye"], self.eos_table["rho"])

        self.eos_table["energy"] = self.eos_table["rho"] * \
            (1 + self.eos_table["eps"])
        self.eos_table["Gamma"] = self.eos_table['energy'] + \
            self.eos_table['press']
        self.eos_table["Gamma"] *= self.eos_table['cs2'] / \
            self.eos_table['press']

        integ = 1/(self.eos_table["energy"] + self.eos_table["press"])
        self.eos_table['enthalpy'] = cumulative_trapezoid(
            integ, self.eos_table['press'], initial=0)

    def _read_pizza_table(self):
        """
        Get the pizza table for EOS.
        """
        self.pizza_table = {}
        pizza_path = self.eos.hydro_path.replace("hydro.h5", "pizza")
        (
            self.pizza_table["rho"],
            self.pizza_table["eps"],
            self.pizza_table["press"],
        ) = np.loadtxt(pizza_path, unpack=True, skiprows=5, usecols=(0, 1, 2))
        self.pizza_table["rho"] *= RUnits["Rho"] / 1e3
        self.pizza_table["press"] *= RUnits["Press"] * 10

        i_table = self.eos_table["rho"].argmin()
        min_rho = self.eos_table["rho"][i_table]
        i_pizza = np.argmin(np.abs(self.pizza_table["rho"] - min_rho))
        eps_offset = (self.pizza_table["eps"]
                      [i_pizza] - self.eos_table["eps"][i_table])
        self.pizza_table["eps"] -= eps_offset

    def _correct_pressure(self):
        """
        Correct any non-monotonic pressure in the EOS table.
        """
        monotonic_mask = np.diff(self.eos_table["press"]) < 0
        while np.any(monotonic_mask):
            i_s = np.where(monotonic_mask)[0][0]
            pickup = self.eos_table["press"][i_s:] > self.eos_table["press"][i_s]
            breakpoint()
            i_e = np.where(pickup)[0][0] + i_s
            if self.verbose:
                print(f"{self}: Correcting non monotonic pressure between ")
                print(f"rho={self.eos_table['rho'][i_s]:.2e} -"
                      f"{self.eos_table['rho'][i_e]:.2e} "
                      f"({i_e - i_s} points)")
            for key, table in self.eos_table.items():
                self.eos_table[key] = np.concatenate(
                    (table[: i_s + 1], table[i_e:]))
            monotonic_mask = np.diff(self.eos_table["press"]) < 0

    def _interpolate_eos_table(
        self,
        value: float,
        key: str,
        x_key: str,
    ):
        """
        Interpolate the EOS table for the given value using the specified keys.

        Args:
            value (float): Value to interpolate.
            key (str): Key to interpolate in the EOS table.
            x_key (str): Key representing the x-coordinate (pressure or
                pseudo enthalpy) for interpolation.

        Returns:
            float: Interpolated value for the given value and key.
        """
        if value <= self.eos_table[x_key][0]:
            return self.eos_table[key][0]

        tabulated = self.eos_table[key]
        log_x = x_key not in _lin_interp_keys
        log_interp = key not in _lin_interp_keys

        x_tabulted = self.eos_table[x_key]
        if log_x:
            x_tabulted = np.log10(x_tabulted)
            value = np.log10(value)
        if log_interp:
            tabulated = np.log10(tabulated)

        res = interp1d(
            x_tabulted,
            tabulated,
            kind="linear",
            bounds_error=True
        )(value)

        if log_interp:
            return 10**res
        return res

    def _get_sources_by_enthalpy(
        self,
        enthalpy: float,
        r_mass_ye: "NDArray[np.float_]",
    ):
        """
        Calculate TOV sources as a function of pseudo enthalpy.
        See Svenja Greifs thesis for more information.

        Args:
            enthalpy (float): Pseudo enthalpy value.
            r_mass_ye (Tuple[float, float, float]): Tuple containing radial coordinate, mass, and electron_fraction.

        Returns:
            np.ndarray: Array containing the derivatives of radial coordinate, mass, and electron_fraction with respect to pseudo enthalpy.
        """
        rr, mass, yy = r_mass_ye

        kw = dict(value=enthalpy, x_key="enthalpy")
        energy = self._interpolate_eos_table(key="energy", **kw)
        cs2 = self._interpolate_eos_table(key="cs2", **kw)
        press = self._interpolate_eos_table(key="press", **kw)

        if rr < 0:
            return np.array([np.nan, np.nan, np.nan])
        fourpir3 = fourpi*rr**3
        rminus2m = rr - 2*mass
        denum = mass + fourpir3*press

        dr = - rr*rminus2m / denum
        dm = -fourpir3 * energy * rminus2m / denum

        dy = rminus2m * (yy + 1) * yy
        dy += (mass - fourpir3*energy) * yy
        dy += fourpir3*(5*energy + 9*press) - 6*rr
        dy += fourpir3*(energy + press) / cs2
        dy /= denum
        dy += yy
        dy -= 4*denum / rminus2m

        drmy = np.array([dr, dm, dy])

        return drmy

    def calculate_offset_values_by_enthalpy(
            self,
            central_enthalpy: float,
            dh: float):
        """
        Get the values of r, mass, and ye at a small offset dh from the central enthalpy.

        Args:
            central_enthalpy (float): Central enthalpy value.
            dh (float): Small offset value from the central enthalpy.

        Returns:
            numpy.ndarray: Array containing the values of r, mass, and ye at the offset enthalpy.
        """

        kw = dict(value=central_enthalpy, x_key="enthalpy")
        Ec = self._interpolate_eos_table(key="energy", **kw)
        cs2 = self._interpolate_eos_table(key="cs2", **kw)
        pc = self._interpolate_eos_table(key="press", **kw)

        eplusp_cs2 = (Ec + pc)/cs2
        eplus3p = Ec + 3 * pc

        r1 = (3 / (2*np.pi * eplus3p))**.5
        r3 = - r1/(4*eplus3p)
        r3 *= Ec - 3*pc - 3*eplusp_cs2 / 5
        m3 = 4*np.pi/3 * Ec * r1**3
        m5 = 4*np.pi * r1**3
        m5 *= r3*Ec/r1 - eplusp_cs2 / 5
        y2 = - 6 / (7*eplus3p)
        y2 *= Ec/3 + 11*pc + eplusp_cs2

        rdh = dh**0.5
        rdh3 = rdh**3
        rdh5 = rdh**5

        rr = r1*rdh + r3*rdh3
        mm = m3*rdh3 * + m5*rdh5
        yy = 2 + y2*dh

        return np.array([rr, mm, yy])

    def solve_by_enthalpy(
        self,
        central_enthalpy: float,
        terminal_enthalpy: Optional[float] = None,
        num_points: int = 1000,
        save: bool = True,
        **solver_kwargs,
    ) -> OdeResult:
        """
        Solve the TOV equations for a given central pseudo enthalpy.

        Args:
            central_enthalpy (float): Central pseudo enthalpy value.
            terminal_enthalpy (float, optional): Terminal pseudo enthalpy value. Default is None.
            num_points (int, optional): Number of points for evaluation. Default is 1000.
            **solver_kwargs: Additional arguments to be passed to the solver.

        Returns:
            scipy.integrate.OdeSolution: The solution of the TOV equations.
        """
        default = dict(method="DOP853", rtol=1e-5, atol=1e-3)
        solver_kwargs = {**default, **solver_kwargs}

        if terminal_enthalpy is None:
            terminal_enthalpy = self.eos_table['enthalpy'][1]

        h_span = central_enthalpy, terminal_enthalpy
        h_eval = np.linspace(
            central_enthalpy,
            terminal_enthalpy,
            num_points
        )

        self.parameters["hcent"] = central_enthalpy

        y0 = self.calculate_offset_values_by_enthalpy(
            central_enthalpy, dh=central_enthalpy*1e-6)

        solution = solve_ivp(
            self._get_sources_by_enthalpy,
            y0=y0,
            t_span=h_span,
            t_eval=h_eval,
            **solver_kwargs
        )

        if solution.status >= 0:
            if save:
                self.data["enthalpy"] = solution.t
                self.data["r"], self.data["m"], self.data["y"] = solution.y
                for val, key in zip((0, 0, 2, central_enthalpy), 'r m y enthalpy'.split()):
                    self.data[key] = np.concatenate(
                        ([val], self.data[key]))
                self.post_process()
                self.is_solved = True

        elif "Required step size is less" in solution.message:
            if solver_kwargs['rtol'] < 1e-1:
                solver_kwargs['rtol'] = solver_kwargs['rtol'] * 2
                if self.verbose:
                    print(
                        f"{self}: Did not converge. Increasing 'rtol' to {solver_kwargs['rtol']:.2e}"
                    )
                return self.solve_by_enthalpy(
                    central_enthalpy=central_enthalpy,
                    num_points=num_points,
                    **solver_kwargs
                )
        else:
            print(f"{self}: {solution.message}")
        return solution

    def _get_sources_by_pressure(self, pressure, r2_mass_ye):
        """
        Calculate TOV sources as a function of pressure.

        Args:
            pressure (float): Pressure value.
            r_squared_mass_ye (Tuple[float, float, float]): Tuple containing radial coordinate squared, mass, and electron_fraction.

        Returns:
            np.ndarray: Array containing the derivatives of radial coordinate squared, mass, and electron_fraction with respect to pressure.
        """
        r2, mass, yy = r2_mass_ye
        if any(map(lambda dd: not np.all(np.isfinite(dd)), r2_mass_ye)):
            return np.array([np.nan, np.nan, np.nan])
        if r2 < 0:
            return np.array([np.nan, np.nan, np.nan])

        kw = dict(value=pressure, x_key="press")
        energy = self._interpolate_eos_table(key="energy", **kw)
        cs2 = self._interpolate_eos_table(key="cs2", **kw)

        r3 = r2**1.5
        rr = r2**.5
        rminus2m = rr - 2*mass
        denum = (energy+pressure)*(mass + fourpi*r3*pressure)
        if denum <= 0.:
            denum = 1e-20

        dr2 = -2 * r2 * rminus2m / denum

        dm = -fourpi * r3 * energy * rminus2m / denum

        F = rr - fourpi * r3 * (energy - pressure)
        F /= rminus2m
        Q = fourpi * rr / rminus2m
        Q *= (5 * energy + 9 * pressure +
              (energy + pressure) / cs2) * r2 - 6 / fourpi
        dnudr = mass + fourpi * r3 * pressure
        dnudr /= rminus2m
        Q -= 4 * dnudr**2

        dy = -(yy**2) - yy * F - Q
        dy /= 2 * r2

        dy *= dr2

        return np.array([dr2, dm, dy])

    def solve_by_pressure(
        self,
        central_pressure: float,
        num_points: int = 1000,
        terminal_pressure: Optional[float] = None,
        save: bool = True,
        **solver_kwargs
    ) -> OdeResult:
        """
        Solve the TOV equations for a given central pressure.
        *** This method is depricated! Use solve_by_enthalpy instead! ***

        Args:
            central_pressure (float): Central pressure value.
            num_points (int, optional): Number of points for evaluation. Default is 1000.
            terminal_pressure (float, optional): Terminal pressure value. Default is None.
            **solver_kwargs: Additional arguments to be passed to the solver.

        Returns:
            scipy.optimize.OdeSolution: The solution of the TOV equations.
        """
        warnings.warn(
            "This method will be depricated. Use 'solve_by_enthalpy' instead.",
            DeprecationWarning
        )

        default = dict(method="DOP853", rtol=1e-5, atol=1e-3)
        solver_kwargs = {**default, **solver_kwargs}

        if terminal_pressure is None:
            terminal_pressure = 1e-17 * Units["Length"]**-2
        p_span = central_pressure, terminal_pressure

        # lin + log grid to make the integration more stable
        p_int = central_pressure*.1
        p_eval = np.concatenate([
            np.linspace(central_pressure, p_int, num_points//2+1)[:-1],
            np.logspace(np.log10(p_int),
                        np.log10(terminal_pressure),
                        num_points//2)
        ])
        p_eval = np.clip(p_eval, terminal_pressure, central_pressure)

        self.parameters["Pcent"] = central_pressure

        y0 = np.array([1e-10, 1e-10, 2.0])

        solution = solve_ivp(
            self._get_sources_by_pressure,
            y0=y0,
            t_span=p_span,
            t_eval=p_eval,
            **solver_kwargs
        )

        if solution.status >= 0:
            if save:
                self.data["press"] = solution.t
                self.data["r"], self.data["m"], self.data["y"] = solution.y
                self.data["r"] = self.data["r"] ** 0.5
                self.post_process()
                self.is_solved = True
        elif "Required step size is less" in solution.message:
            if solver_kwargs['rtol'] < 1e-1:
                solver_kwargs['rtol'] = solver_kwargs['rtol'] * 2
                if self.verbose:
                    print(
                        f"{self}: Did not converge. Increasing 'rtol' to {solver_kwargs['rtol']:.2e}"
                    )
                return self.solve_by_pressure(
                    central_press=central_pressure,
                    num_points=num_points,
                    terminal_pressure=terminal_pressure,
                    **solver_kwargs
                )

        else:
            print(f"{self}: {solution.message}")
        return solution

    def post_process(self):
        """
        Perform post-processing after solving the TOV equations.
        """
        self.data["C"] = np.zeros_like(self.data["r"])
        self.data["C"][1:] = self.data["m"][1:] / self.data["r"][1:]
        self.parameters["R"] = self.data["r"][-1]
        self.parameters["M"] = self.data["m"][-1]
        self.parameters["yR"] = self.data["y"][-1]
        self.parameters["C"] = self.parameters["M"] / self.parameters["R"]
        y = self.parameters["yR"]
        c = self.parameters["C"]
        # Eq. (23) in Hinderer etal (2008)
        # (note that there is an Erratum for this equation)
        self.parameters["k_2"] = (
            8/5 * c**5 * (1 - 2 * c) ** 2 * (2 + 2 * c * (y - 1) - y)
            / (
                2 * c * (6 - 3 * y + 3 * c * (5 * y - 8))
                + 4 * c**3
                * (13 - 11 * y + c * (3 * y - 2) + 2 * c**2 * (1 + y))
                + 3 * (1 - 2 * c) ** 2
                * (2 - y + 2 * c * (y - 1)) * np.log(1 - 2 * c)
            )
        )
        k2C = self.parameters["k_2"] * self.parameters["C"]**-5
        self.parameters["Lambda"] = 2/3 * k2C
        self.parameters["kappa2T"] = 1/8 * k2C
        if 'press' in self.data:
            for key in "rho eps cs2 energy ye enthalpy".split():
                self.data[key] = np.array(
                    [self._interpolate_eos_table(pp, key, x_key="press")
                     for pp in self.data["press"]]
                )
        elif 'enthalpy' in self.data:
            for key in "rho eps cs2 energy ye press".split():
                self.data[key] = np.array(
                    [self._interpolate_eos_table(hh, key, x_key="enthalpy")
                     for hh in self.data["enthalpy"]]
                )

    def find_central_value_for_target_mass(
        self,
        target_mass: float,
        root_finder: Callable = bisect,
        solver_kwargs: Dict = {},
        solve_by: str = "enthalpy",
        **kwargs
    ) -> float:
        """
        Find the central value for a given target mass.
        The value can either be enthalpy or pressure depending on the solve_by keyword argument.

        Args:
            target_mass (float): Target mass value.
            root_finder (callable, optional): The root-finding function to use. Default is scipy.optimize.bisect.
            solver_kwargs (dict, optional): Additional arguments to be passed to the solver function.
            solve_by (str, optional): Which solve method should be used, either "enthalpy" or "pressure" (default: "enthalpy")
            **kwargs: Additional arguments to be passed to the root-finding function.

        Returns:
            float: The central pressure corresponding to the target mass.
        """

        verbose = self.verbose

        if verbose:
            print(f"{self}: Finding central {solve_by} for {target_mass}M star")
            print(f"central {solve_by:>8s} {'mass':>11s} {'radius':>11s}")
            self.verbose = False

        if solve_by == 'enthalpy':
            solver = self.solve_by_enthalpy
            a = .1
        elif solve_by == 'pressure':
            solver = self.solve_by_pressure
            a = 5e-5
        else:
            raise ValueError(
                f"Unknown solve_by method: {solve_by}."
                "Must be either 'enthalpy' or 'pressure'."
            )

        b_max = self.eos_table[solve_by].max()
        default = dict(a=a, b=b_max, rtol=1e-5, maxiter=100)
        kwargs = {**default, **kwargs}

        def _get_mass(cental_value):
            solution = solver(cental_value, save=False, **solver_kwargs)
            if solution.status < 0:
                return target_mass
            if verbose:
                print(
                    f"{cental_value:16.6e} {solution.y[1][-1]:11.8f} "
                    f"{solution.y[0][-1]:11.8f}"
                )
            return solution.y[1][-1] - target_mass

        central_value = root_finder(_get_mass, **kwargs)

        self.verbose = verbose
        solver(central_value, **solver_kwargs)

        return central_value

    def find_maximum_mass(
        self,
        min_finder: Callable = minimize_scalar,
        solver_kwargs: Dict = {},
        solve_by: str = "enthalpy",
        **kwargs
    ) -> float:
        """
        Find the maximum TOV mass for the EOS.

        Args:
            min_finder (callable, optional): The root-finding function to use. Default is scipy.optimize.bisect.
            solver_kwargs (dict, optional): Additional arguments to be passed to the solver function.
            solve_by (str, optional): Which solve method should be used, either "enthalpy" or "pressure" (default: "enthalpy")
            **kwargs: Additional arguments to be passed to the maximum-finding function.

        Returns:
            float: The central pressure corresponding to the target mass.
        """
        verbose = self.verbose

        if verbose:
            print(
                f"{self}: Finding maximum TOV mass by solving for {solve_by}")
            print(f"central {solve_by:>8s} {'mass':>11s} {'radius':>11s}")
            self.verbose = False

        if solve_by == 'enthalpy':
            solver = self.solve_by_enthalpy
            a = .1
        elif solve_by == 'pressure':
            solver = self.solve_by_pressure
            a = 5e-5
        else:
            raise ValueError(
                f"Unknown solve_by method: {solve_by}."
                "Must be either 'enthalpy' or 'pressure'."
            )

        v_max = self.eos_table[solve_by].max()
        default = dict(bracket=(a, v_max*.3))
        kwargs = {**default, **kwargs}

        def _get_mass(cental_value):
            solution = solver(cental_value, save=False, **solver_kwargs)
            if solution.status < 0:
                return 0
            if verbose:
                print(
                    f"{cental_value:16.6e} {solution.y[1][-1]:11.8f} "
                    f"{solution.y[0][-1]:11.8f}"
                )
            return -solution.y[1][-1]

        result = min_finder(_get_mass, **kwargs)
        central_value = result.x

        self.verbose = verbose
        solver(central_value, **solver_kwargs)

        return self.parameters["M"]

    def __str__(self):
        """
        Get a string representation of the TOV object.

        Returns:
            str: String representation of the TOV object.
        """
        if self.is_solved:
            return f"TOV (M={self.parameters['M']:.2f}M, R={self.parameters['R']:.2f}km) EOS: {self.eos}"
        return f"TOV (unsolved) EOS: {self.eos}"

    def __repr__(self):
        """
        Get a string representation of the TOV object.

        Returns:
            str: String representation of the TOV object.
        """
        return self.__str__()
