"""TODO use an actuall rc file and parse it"""

import os
from typing import Type, List
from . import Utils
from .DataHandlers import PackET2Handler, DataHandler, PackETHandler


notable_dir = os.path.dirname(Utils.__file__)

class Config():
    """Holds default parameters"""
    default_data_handler: Type[DataHandler] = PackETHandler
    GridFuncVariable_json: str = f"{notable_dir}/GridFuncVariables.json"
    TimeSeriesVariables_json: str = f"{notable_dir}/TimeSeriesVariables.json"
    PPGridFuncVariable_files: List[str] = [f"{notable_dir}/PPHydroVariables.py",
                                           f"{notable_dir}/PPSpacetimeVariables.py",
                                           f"{notable_dir}/PPEOSVariables.py"]
    PPTimeSeries_files: List[str] = [f"{notable_dir}/PPTimeSeriesVariables.py"]
    PPGW_files: List[str] = [f"{notable_dir}/PPGWVariables.py"]
    surf_int_n_theta: int = 50
    surf_int_n_phi: int = 100


config = Config()
