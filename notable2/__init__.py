"""
notable2 (working title)
A bunch of modular postprocessing and plotting scripts.
"""
from collections import namedtuple

from matplotlib.colors import ColorConverter

from .Simulation import Simulation
from .Animations import Animation
from .EOS import TabulatedEOS
from .TOV import TOVSolver
from .Utils import Units, RUnits


# create a class from the dictionary of units for easy access
lower_U = {k.lower(): v for k, v in Units.items()}
lower_RU = {k.lower(): v for k, v in RUnits.items()}
U = namedtuple('Units', lower_U.keys())(**lower_U)
RU = namedtuple('RUnits', lower_RU.keys())(**lower_RU)
