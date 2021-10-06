from abc import ABC, abstractmethod


class EOS(ABC):
    """EOS abstract base class"""
    ...


class TabulateEOS(EOS):
    """Realistic Tabluated EOS """

    def __init__(self, path):
        self.path = path
        ...
