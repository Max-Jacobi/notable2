"""TODO use an actuall rc file and parse it"""

import os
from typing import Type
from dataclasses import dataclass
from . import Utils
from .DataHandlers import PackETGetter, DataHandler


home = os.environ["HOME"]
notable_dir = os.path.dirname(Utils.__file__)
if (xdg_config_home := os.environ["XDG_CONFIG_HOME"]) == '':
    if home != '':
        xdg_config_home = f'{home}/.config'


@dataclass
class RCParams():
    """Holds default parameters"""
    default_getter: Type[DataHandler] = PackETGetter
    default_eos_path: str = f'{home}/desert/simulations/EOS/DD2'
    GridVariable_json: str = f'{notable_dir}/GridVariables.json'
    TimeSeriesVariables_json: str = f'{notable_dir}/TimeSeriesVariables.json'


rcParams = RCParams()
