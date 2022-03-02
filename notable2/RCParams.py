"""TODO use an actuall rc file and parse it"""

import os
from typing import Type
from dataclasses import dataclass
from . import Utils
from .DataHandlers import PackET2Handler, DataHandler


home = os.environ["HOME"]
notable_dir = os.path.dirname(Utils.__file__)
if (xdg_config_home := os.environ["XDG_CONFIG_HOME"]) == "":
    if home != "":
        xdg_config_home = f"{home}/.config"


class RCParams():
    """Holds default parameters"""
    default_data_handler: Type[DataHandler] = PackET2Handler
    default_eos_path: str = f"{home}/desert/simulations/EOS/LS220/hydro.h5"
    GridFuncVariable_json: str = f"{notable_dir}/GridFuncVariables.json"
    TimeSeriesVariables_json: str = f"{notable_dir}/TimeSeriesVariables.json"
    TimeSeriesVariables_json: str = f"{notable_dir}/TimeSeriesVariables.json"
    PPGridFuncVariable_files: list[str] = [f"{notable_dir}/PPHydroVariables.py",
                                           f"{notable_dir}/PPSpacetimeVariables.py",
                                           f"{notable_dir}/PPEOSVariables.py"]
    PPTimeSeries_files: list[str] = [f"{notable_dir}/PPTimeSeriesVariables.py"]
    PPGW_files: list[str] = [f"{notable_dir}/PPGWVariables.py"]
    surf_int_n_theta: int = 50
    surf_int_n_phi: int = 100


rcParams = RCParams()
