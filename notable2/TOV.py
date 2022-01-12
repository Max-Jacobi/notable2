from typing import TYPE_CHECKING, Optional
import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import interp1d
from scipy.optimize import toms748
from scipy.integrate import solve_ivp

from .Utils import RUnits

if TYPE_CHECKING:
    from .EOS import TabulatedEOS


class TOV:
    table: dict[str, NDArray[np.float_]]
    pizza_table: dict[str, NDArray[np.float_]]
    data: dict[str, NDArray[np.float_]]
    prameters: dict[str, float]

    def __init__(self,
                 eos: "TabulatedEOS",
                 verbose: bool = False,
                 save: bool = True):

        self.parameters = {}
        self.pizza_table = {}
        self.data = {"r": np.array([]),
                     "m": np.array([]),
                     "p": np.array([]),
                     "y": np.array([])}
        self.eos = eos
        self.verbose = verbose
        self.save = save
        self._get_table()
        self._correct_press()

    def _get_table(self):
        self.table = {}
        self.table["rho"] = self.eos.get_key('density')
        self.table["ye"] = np.zeros_like(self.table["rho"])
        ye_0 = self.eos.get_key('ye')
        rho, ye = np.meshgrid(self.table["rho"], ye_0, indexing='ij')
        getter = self.eos.get_cold_caller(['mu_p', 'mu_e', 'mu_n'],
                                          lambda mu_p, mu_e, mu_n, *_: mu_p + mu_e - mu_n)
        mu_nu = getter(ye, rho)

        for ii, mn in enumerate(mu_nu):
            f = interp1d(ye_0, mn, kind='cubic', bounds_error=True)
            try:
                self.table["ye"][ii] = toms748(f, a=ye_0[0], b=ye_0[-1], k=2,
                                               xtol=1e-6, rtol=1e-3, maxiter=100)
            except ValueError:
                self.table["ye"][ii] = ye_0.min()

        self.table["mu_nu"] = getter(self.table["ye"], self.table["rho"])
        getter = self.eos.get_cold_caller(['pressure'])
        self.table["press"] = getter(self.table["ye"], self.table["rho"])
        getter = self.eos.get_cold_caller(['internalEnergy'])
        self.table["eps"] = getter(self.table["ye"], self.table["rho"])
        getter = self.eos.get_cold_caller(['cs2'])
        self.table["cs2"] = getter(self.table["ye"], self.table["rho"])

        self.table["En"] = self.table["rho"] * (1 + self.table["eps"])

    def get_pizza(self):
        self.pizza_table = {}
        pizza_path = self.eos.hydro_path.replace("hydro.h5", "pizza")
        self.pizza_table["rho"], self.pizza_table["eps"], self.pizza_table["press"] = np.loadtxt(
            pizza_path,
            unpack=True,
            skiprows=5,
            usecols=(0, 1, 2)
        )
        self.pizza_table["rho"] *= RUnits['Rho']/1e3
        self.pizza_table["press"] *= RUnits['Press']*10

        i_table = self.table["rho"].argmin()
        min_rho = self.table["rho"][i_table]
        i_pizza = np.argmin(np.abs(self.pizza_table["rho"] - min_rho))
        eps_offset = self.pizza_table["eps"][i_pizza] - self.table["eps"][i_table]
        self.pizza_table["eps"] -= eps_offset

    def _correct_press(self):
        mon_mask = np.diff(self.table["press"]) < 0
        while np.any(mon_mask):
            i_s = np.where(mon_mask)[0][0]
            pickup = self.table["press"][i_s:] > self.table["press"][i_s]
            i_e = np.where(pickup)[0][0] + i_s
            print('correcting for non monotonic pressure '
                  f'at rho={self.table["rho"][i_s]:.2e} - {self.table["rho"][i_e]:.2e} '
                  f'({i_e - i_s} points)')
            for key, table in self.table.items():
                self.table[key] = np.concatenate((table[:i_s+1], table[i_e:]))
            mon_mask = np.diff(self.table["press"]) < 0

    def _call_eos(self, press, key):
        if press <= self.table["press"][0]:
            return self.table[key][0]
        return interp1d(self.table["press"], self.table[key], kind='cubic', bounds_error=True)(press)

    def _get_sources(self, rad, mpy):
        mass, press, yy = mpy
        En = self._call_eos(press, "En")
        rho = self._call_eos(press, "rho")
        cs2 = self._call_eos(press, "cs2")

        fourpi = 4*np.pi
        rminus2m = rad - 2*mass

        dP = -press - En
        dP *= mass + fourpi * rad**3 * press
        dP /= rminus2m * rad

        dm = fourpi * rad**2 * rho

        F = rad - fourpi * rad**3 * (En - press)
        F /= rminus2m
        Q = fourpi * rad / rminus2m
        Q *= (5*En + 9*press + (En + press)/cs2)*rad**2 - 6/fourpi
        dnudr = mass + fourpi * rad**3 * press
        dnudr /= rminus2m
        Q -= 4*dnudr**2

        dy = -yy**2 - yy*F - Q
        dy /= rad

        return np.array([dm, dP, dy])

    def solve(self, central_press, dr_out=.01, terminal_pressure=None, **kwargs):
        if terminal_pressure is None:
            terminal_pressure = self.table["press"][0]
        t_span = 1e-14, 50
        N = int(50/dr_out)+1
        t_eval = np.linspace(*t_span, N)
        y0 = np.array([0, central_press, 2.])

        def terminate(_, mpy):
            return mpy[1] - terminal_pressure
        terminate.terminal = True

        solution = solve_ivp(self._get_sources,
                             y0=y0,
                             t_span=t_span,
                             t_eval=t_eval,
                             events=terminate,
                             **kwargs)
        if solution.status >= 0:
            self.data['r'] = solution.t
            self.data['m'], self.data['p'], self.data['y'] = solution.y
            self.post_proc()
        if solution.status == 1:
            self.parameters["R"] = solution.t_events[0][0]
            self.parameters["M"] = solution.y_events[0][0, 0]
            self.parameters["yR"] = solution.y_events[0][0, 2]
            self.parameters["C"] = self.parameters["M"]/self.parameters["R"]
            y = self.parameters["yR"]
            c = self.parameters["C"]
            self.parameters["k_2"] = (
                8/5*c**5 * (1-2*c)**2 * (2 + 2*c*(y-1) - y)
                / (2*c*(6 - 3*y + 3*c*(5*y - 8))
                   + 4*c**3 * (13 - 11*y + c*(3*y - 2) + 2*c**2 * (1 + y))
                   + 3*(1 - 2*c)**2 * (2 - y + 2*c*(y - 1)) * np.log(1 - 2*c)))
        return solution.status

    def post_proc(self):
        self.data["C"] = self.data["m"] / self.data["r"]
        y = self.data["y"][1:]
        c = self.data["C"][1:]
        self.data["k_2"] = np.zeros_like(self.data["r"])
        self.data["k_2"][1:] = (8/5*c**5 * (1-2*c)**2 * (2 + 2*c*(y-1) - y)
                                / (2*c*(6 - 3*y + 3*c*(5*y - 8))
                                   + 4*c**3 * (13 - 11*y + c*(3*y - 2) + 2*c**2 * (1 + y))
                                   + 3*(1 - 2*c)**2 * (2 - y + 2*c*(y - 1)) * np.log(1 - 2*c)))
        for key in "rho eps cs2 En ye".split():
            self.data[key] = np.array([self._call_eos(pp, key) for pp in self.data["p"]])

    def get_central_p(self, m_target, a=5e-5, b=1e-3):
        verbose = self.verbose
        save = self.save
        self.save = False
        if verbose:
            print(f"Finding central pressure for {m_target}M star")
            print(f"{'central pressure':>16s} {'mass':>11s} {'radius':>11s}")
            self.verbose = False

        def _get_mass(p_cent):
            sol = self.solve(p_cent, dr_out=1e-2, rtol=1e-8, method="DOP853")
            if sol < 0:
                raise RuntimeError(f"solver did not converge for central pressure {p_cent}")
            if sol == 0:
                raise RuntimeError(f"solver reached r=50M for central pressure {p_cent}")
            if verbose:
                print(f"{p_cent:16.6e} {self.parameters['M']:11.8f} {self.parameters['R']:11.8f}")
            return self.parameters["M"] - m_target

        p_cent = toms748(_get_mass, a=a, b=b, k=2, rtol=1e-5, maxiter=100)
        self.verbose = verbose
        self.save = save
        return p_cent