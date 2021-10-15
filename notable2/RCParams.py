"""TODO use an actuall rc file and parse it"""

import os
from typing import Type
from dataclasses import dataclass
from . import Utils
from .DataHandlers import PackETHandler, DataHandler


home = os.environ["HOME"]
notable_dir = os.path.dirname(Utils.__file__)
if (xdg_config_home := os.environ["XDG_CONFIG_HOME"]) == '':
    if home != '':
        xdg_config_home = f'{home}/.config'


class RCParams():
    """Holds default parameters"""
    default_data_handler: Type[DataHandler] = PackETHandler
    default_eos_path: str = f'{home}/desert/simulations/EOS/DD2'
    GridDataVariable_json: str = f'{notable_dir}/GridDataVariables.json'
    TimeSeriesVariables_json: str = f'{notable_dir}/TimeSeriesVariables.json'
    UGridDataVariable_files: list[str] = [f'{notable_dir}/UDHydroVariables.py', f'{notable_dir}/UDSpacetimeVariables.py']
    UTimeSeries_files: list[str] = [f'{notable_dir}/UDTimeSeriesVariables.py']
    surf_int_n_theta: int = 50
    surf_int_n_phi: int = 100


rcParams = RCParams()
