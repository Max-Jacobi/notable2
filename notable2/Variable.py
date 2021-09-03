from abc import ABC, abstractmethod

from .Utils import Simulation


class Variable(ABC):
    """Documentation for Variable

    """
    sim: Simulation
    key: str

    def __init__(self, key: str, sim: Simulation, ):
        self.sim = sim
        self.key = key
        ...
